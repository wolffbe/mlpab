"""Treatment session orchestration after the build barrier was removed.

Setup is a single up-front phase: preflight builds + tests each platform AND
materializes its prepared venv (interfaces.prepare). Every run then just clones
its prepared venv read-only, so there is no per-run build and no barrier — runs
are no longer wrapped in an agent slot. These tests assert the surviving
contract: setup runs ONCE before any run, and every configured run executes.
"""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mlpab import claude_runner, preflight, results, runner, treatments


class SetupPhaseTests(unittest.TestCase):
    def test_preflight_runs_once_before_every_run(self):
        events = []

        cfg = treatments.TreatmentConfig(
            runs=[
                treatments.RunEntry(
                    task="training_data", platform="hopsworks", interface="cli", category="feature"
                ),
                treatments.RunEntry(
                    task="training_data", platform="hopsworks", interface="cli", category="feature2"
                ),
            ],
        )
        tmp = Path(tempfile.mkdtemp())

        def fake_run(spec):
            events.append("run")
            # Stop before per-run post-processing (results row, notebook): the
            # loop catches this and records a failure, which is fine here.
            raise RuntimeError("stop after run")

        with (
            mock.patch.object(preflight, "check_availability", lambda *a, **k: None),
            mock.patch.object(treatments, "check_readiness", lambda cfg: (True, [])),
            mock.patch.object(treatments, "check_llm_live", lambda *a, **k: (True, [])),
            mock.patch.object(preflight, "preflight", lambda *a, **k: events.append("preflight")),
            mock.patch.object(claude_runner, "oauth_token_from_keychain", lambda: None),
            mock.patch.object(results, "roll_up_results", lambda *a, **k: None),
            mock.patch.object(runner, "run", side_effect=fake_run),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            treatments.run_treatments(cfg, tmp, config_name="t")

        # Setup happened exactly once, before any run; then every run executed.
        self.assertEqual(events.count("preflight"), 1)
        self.assertEqual(events[0], "preflight")
        self.assertEqual(events.count("run"), 2)
        # No barrier wrappers survive on the module.
        self.assertFalse(hasattr(preflight, "agent_slot"))


class SetupVerifyAbortsConfigTests(unittest.TestCase):
    def test_platform_not_ready_aborts_whole_config(self):
        calls = []

        def fake_run(spec):
            calls.append(spec)
            raise preflight.PlatformNotReadyError("databricks not ready")

        cfg = treatments.TreatmentConfig(
            runs=[
                treatments.RunEntry(
                    task="training_data", platform="databricks", interface="cli", category="feature"
                ),
                treatments.RunEntry(
                    task="training_data",
                    platform="databricks",
                    interface="cli",
                    category="feature2",
                ),
            ],
        )
        tmp = Path(tempfile.mkdtemp())
        with (
            mock.patch.object(preflight, "check_availability", lambda *a, **k: None),
            mock.patch.object(treatments, "check_readiness", lambda cfg: (True, [])),
            mock.patch.object(treatments, "check_llm_live", lambda *a, **k: (True, [])),
            mock.patch.object(preflight, "preflight", lambda *a, **k: None),
            mock.patch.object(claude_runner, "oauth_token_from_keychain", lambda: None),
            mock.patch.object(results, "roll_up_results", lambda *a, **k: None),
            mock.patch.object(runner, "run", side_effect=fake_run),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            with self.assertRaises(preflight.PlatformNotReadyError):
                treatments.run_treatments(cfg, tmp, config_name="t")
        # Aborted on the FIRST run — never attempted the second.
        self.assertEqual(len(calls), 1)


class LivenessAbortsConfigTests(unittest.TestCase):
    def test_unreachable_model_aborts_before_any_run(self):
        # A model that passes static readiness but fails the live probe must
        # abort the session before any run (prevents zero-activity runs).
        ran = []
        cfg = treatments.TreatmentConfig(
            runs=[
                treatments.RunEntry(
                    task="training_data", platform="databricks", interface="cli", category="feature"
                )
            ],
        )
        with (
            mock.patch.object(preflight, "check_availability", lambda *a, **k: None),
            mock.patch.object(treatments, "check_readiness", lambda cfg: (True, [])),
            mock.patch.object(
                treatments,
                "check_llm_live",
                lambda *a, **k: (False, [("mistral-medium-3.5", False, "exit 1: model not found")]),
            ),
            mock.patch.object(
                preflight,
                "preflight",
                mock.Mock(side_effect=AssertionError("must not reach build")),
            ),
            mock.patch.object(runner, "run", mock.Mock(side_effect=lambda s: ran.append(s))),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            with self.assertRaises(preflight.PreflightError):
                treatments.run_treatments(cfg, Path(tempfile.mkdtemp()), config_name="t")
        self.assertEqual(ran, [])


class UnimplementedFamilyFailFastTests(unittest.TestCase):
    def test_unknown_eval_family_aborts_before_any_setup(self):
        # run_treatments validates every task against the evals registry
        # BEFORE building anything — an unimplemented family must raise.
        cfg = treatments.TreatmentConfig(
            runs=[treatments.RunEntry(task="no-such-eval", platform="hopsworks", interface="cli")],
        )
        with (
            mock.patch.object(preflight, "check_availability", lambda *a, **k: None),
            mock.patch.object(treatments, "check_readiness", lambda cfg: (True, [])),
            mock.patch.object(
                preflight,
                "preflight",
                mock.Mock(side_effect=AssertionError("must not reach preflight")),
            ),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            with self.assertRaises(ValueError):
                treatments.run_treatments(cfg, Path(tempfile.mkdtemp()), config_name="t")


if __name__ == "__main__":
    unittest.main()
