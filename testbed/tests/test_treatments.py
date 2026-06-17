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


class _SyncExecutor:
    """A ProcessPoolExecutor stand-in that runs submit() synchronously in-process
    so a mocked ``runner.run`` is exercised (a real pool would spawn workers that
    never see the parent's mock)."""

    def __init__(self, max_workers=None, initializer=None):
        self._init = initializer

    def __enter__(self):
        if self._init:
            self._init()
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, fn, *args):
        import concurrent.futures as cf

        fut = cf.Future()
        try:
            fut.set_result(fn(*args))
        except Exception as e:  # noqa: BLE001
            fut.set_exception(e)
        return fut


def _fake_row(**kw):
    import types

    base = dict(
        task="training_data",
        platform="hopsworks",
        interface="cli",
        skills="none",
        asserts_passed=1,
        total_asserts=1,
        total_tokens=10,
        wall_time_s=1.0,
        cost_usd=0.01,
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


def _patches(run_side_effect):
    return [
        mock.patch.object(preflight, "check_availability", lambda *a, **k: None),
        mock.patch.object(treatments, "check_readiness", lambda cfg: (True, [])),
        mock.patch.object(treatments, "check_llm_live", lambda *a, **k: (True, [])),
        mock.patch.object(preflight, "preflight", lambda *a, **k: None),
        mock.patch.object(claude_runner, "oauth_token_from_keychain", lambda: None),
        mock.patch.object(results, "roll_up_results", lambda *a, **k: None),
        mock.patch.object(treatments, "_refresh_notebook", lambda *a, **k: None),
        mock.patch.object(runner, "run", side_effect=run_side_effect),
    ]


class ConcurrencyConfigTests(unittest.TestCase):
    def _write(self, body: str) -> Path:
        p = Path(tempfile.mkdtemp()) / "t.yaml"
        p.write_text(body)
        return p

    def test_default_concurrency_is_one(self):
        cfg = treatments.load_config(
            self._write("model: claude-opus-4-8\nruns:\n  - {task: training_data, platform: none, interface: none}\n")
        )
        self.assertEqual(cfg.concurrency, 1)

    def test_concurrency_key_parsed(self):
        cfg = treatments.load_config(
            self._write("model: claude-opus-4-8\nconcurrency: 3\nruns:\n  - {task: training_data, platform: none, interface: none}\n")
        )
        self.assertEqual(cfg.concurrency, 3)

    def test_parallel_alias_and_floor(self):
        cfg = treatments.load_config(
            self._write("model: claude-opus-4-8\nparallel: 0\nruns:\n  - {task: training_data, platform: none, interface: none}\n")
        )
        self.assertEqual(cfg.concurrency, 1)  # clamped to >= 1


class ConcurrentDispatchTests(unittest.TestCase):
    def test_all_runs_execute_in_pool(self):
        seen = []

        def fake_run(spec):
            seen.append((spec.platform, spec.interface, spec.category))
            return _fake_row()

        cfg = treatments.TreatmentConfig(
            runs=[
                treatments.RunEntry(task="training_data", platform="hopsworks", interface="cli", category="feature"),
                treatments.RunEntry(task="training_data", platform="hopsworks", interface="sdk", category="feature"),
            ],
            concurrency=2,
        )
        with mock.patch("concurrent.futures.ProcessPoolExecutor", _SyncExecutor):
            with contextlib.ExitStack() as stack:
                for p in _patches(fake_run):
                    stack.enter_context(p)
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
                treatments.run_treatments(cfg, Path(tempfile.mkdtemp()), config_name="t")
        self.assertEqual(len(seen), 2)
        self.assertIn(("hopsworks", "cli", "feature"), seen)
        self.assertIn(("hopsworks", "sdk", "feature"), seen)

    def test_repeats_get_distinct_attempts(self):
        attempts = []

        def fake_run(spec):
            attempts.append(spec.attempt)
            return _fake_row()

        # One combo, 3 repeats, run concurrently — each must get its own /<n>.
        cfg = treatments.TreatmentConfig(
            runs=[treatments.RunEntry(task="training_data", platform="hopsworks", interface="cli", category="feature")],
            repeats=3,
            concurrency=3,
        )
        with mock.patch("concurrent.futures.ProcessPoolExecutor", _SyncExecutor):
            with contextlib.ExitStack() as stack:
                for p in _patches(fake_run):
                    stack.enter_context(p)
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
                treatments.run_treatments(cfg, Path(tempfile.mkdtemp()), config_name="t")
        self.assertEqual(sorted(attempts), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
