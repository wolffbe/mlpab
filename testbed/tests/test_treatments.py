"""Treatment session orchestration around the cross-session build barrier.

The barrier window must cover the WHOLE setup phase — platform preflight and
interface builds — not just the build: parallel rq1 sessions otherwise sail
past the agent gate while a sibling is still setting up, and open their
agent (claude) mid-setup. Each agent run must also HOLD its slot
(`agent_slot`) for the run's duration, so a sibling session started after
the agent opened waits instead of building under it.
"""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from banter import treatments, claude_runner, preflight, results, runner


class SetupBarrierWindowTests(unittest.TestCase):
    def test_setup_phase_holds_barrier_and_every_run_gates_on_it(self):
        events = []
        state = {"building": False, "slot": False}

        @contextlib.contextmanager
        def fake_building(lock_path=None):
            state["building"] = True
            events.append("building-enter")
            try:
                yield
            finally:
                state["building"] = False
                events.append("building-exit")

        @contextlib.contextmanager
        def fake_slot(lock_path=None, run_lock_path=None):
            events.append(("slot-enter", state["building"]))
            state["slot"] = True
            try:
                yield 0.0
            finally:
                state["slot"] = False
                events.append("slot-exit")

        cfg = treatments.TreatmentConfig(
            runs=[
                treatments.RunEntry(task="training_data", platform="hopsworks",
                                   interface="cli", category="feature"),
                treatments.RunEntry(task="training_data", platform="hopsworks",
                                   interface="cli", category="feature2"),
            ],
        )
        tmp = Path(tempfile.mkdtemp())

        with mock.patch.object(preflight, "building", fake_building), \
             mock.patch.object(preflight, "agent_slot", fake_slot), \
             mock.patch.object(preflight, "preflight",
                               lambda *a, **k: events.append(("preflight", state["building"]))), \
             mock.patch.object(claude_runner, "oauth_token_from_keychain", lambda: None), \
             mock.patch.object(results, "roll_up_results", lambda *a, **k: None), \
             mock.patch.object(runner, "run", mock.Mock(
                 side_effect=lambda spec: events.append(("run", state["building"], state["slot"]))
                 or (_ for _ in ()).throw(RuntimeError("stop after spawn point")))), \
             contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            treatments.run_treatments(cfg, tmp, config_name="t")

        # The whole setup phase ran INSIDE the barrier window.
        self.assertIn(("preflight", True), events)

        # The window closed before any agent run; every run start gated on
        # (and then HELD) its agent slot for the run's duration.
        exit_idx = events.index("building-exit")
        run_idxs = [i for i, e in enumerate(events) if e == ("run", False, True)]
        slot_idxs = [i for i, e in enumerate(events) if e == ("slot-enter", False)]
        self.assertEqual(len(run_idxs), 2)
        self.assertEqual(len(slot_idxs), 2)
        for s, r in zip(slot_idxs, run_idxs):
            self.assertGreater(s, exit_idx)
            self.assertLess(s, r)
        # Slots were released between runs (held per run, not for the session).
        self.assertEqual(events.count("slot-exit"), 2)
        # No agent ever ran while this session's own barrier was held, and
        # none ran outside a held slot.
        self.assertNotIn(("run", True, True), events)
        self.assertNotIn(("run", False, False), events)


class UnimplementedFamilyFailFastTests(unittest.TestCase):
    def test_unknown_eval_family_aborts_before_any_setup(self):
        # run_treatments validates every task against the evals registry
        # BEFORE building anything — an unimplemented family must raise.
        cfg = treatments.TreatmentConfig(
            runs=[treatments.RunEntry(task="no-such-eval", platform="hopsworks",
                                     interface="cli")],
        )
        with mock.patch.object(preflight, "preflight",
                               mock.Mock(side_effect=AssertionError("must not reach preflight"))), \
             contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(ValueError):
                treatments.run_treatments(cfg, Path(tempfile.mkdtemp()), config_name="t")


if __name__ == "__main__":
    unittest.main()
