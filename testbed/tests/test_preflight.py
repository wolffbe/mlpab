"""Cross-session build barrier (preflight.building / preflight.agent_slot).

Parallel treatment sessions (the per-platform rq1 configs) must not START an
agent run while a sibling session is still building its platform — the
no-build platform (sagemaker) otherwise opens claude mid-build of another.
The reverse holds too: a session started AFTER an agent opened (staggered
start) must wait in `building()` until the open run finishes, instead of
building under it.
"""
import multiprocessing
import tempfile
import threading
import time
import unittest
from pathlib import Path

from banter import preflight


def _hold_build_lock(path: str, hold_s: float, started) -> None:
    with preflight.building(Path(path)):
        started.set()
        time.sleep(hold_s)


def _enter_shared(path: str, entered, release) -> None:
    with preflight.building(Path(path)):
        entered.set()
        release.wait(10)


def _hold_agent_slot(path: str, started, release) -> None:
    with preflight.agent_slot(Path(path)):
        started.set()
        release.wait(10)


def _agent_slot_wait(path: str, queue) -> None:
    with preflight.agent_slot(Path(path)) as waited:
        queue.put(waited)


class BuildBarrierTests(unittest.TestCase):
    def setUp(self):
        self.lock = Path(tempfile.mkdtemp()) / ".build-barrier.lock"

    def test_no_holder_returns_immediately(self):
        # Barrier file not created yet (no session ever built) — zero wait.
        self.assertEqual(preflight.await_builds(self.lock), 0.0)
        # File exists but nobody holds it — near-zero wait.
        self.lock.touch()
        self.assertLess(preflight.await_builds(self.lock), 1.0)

    def test_agent_waits_for_parallel_build(self):
        # A sibling session is mid-build (shared lock held): await_builds must
        # block until it releases, not sail through and open the agent.
        ctx = multiprocessing.get_context("spawn")
        started = ctx.Event()
        p = ctx.Process(target=_hold_build_lock, args=(str(self.lock), 2.0, started))
        p.start()
        try:
            self.assertTrue(started.wait(30), "builder never acquired the lock")
            waited = preflight.await_builds(self.lock)
        finally:
            p.join(30)
        self.assertEqual(p.exitcode, 0)
        self.assertGreater(waited, 0.5)

    def test_parallel_builds_share_the_barrier(self):
        # The barrier must not serialize BUILDS — two sessions hold the shared
        # side simultaneously; only agent starts take the exclusive side.
        ctx = multiprocessing.get_context("spawn")
        entered1, entered2, release = ctx.Event(), ctx.Event(), ctx.Event()
        p1 = ctx.Process(target=_enter_shared, args=(str(self.lock), entered1, release))
        p2 = ctx.Process(target=_enter_shared, args=(str(self.lock), entered2, release))
        p1.start()
        p2.start()
        try:
            self.assertTrue(entered1.wait(30))
            self.assertTrue(entered2.wait(30))
        finally:
            release.set()
            p1.join(30)
            p2.join(30)
        self.assertEqual(p1.exitcode, 0)
        self.assertEqual(p2.exitcode, 0)

    def test_agent_slot_acquires_immediately_without_setup(self):
        # No session is setting up — the slot is free and the wait is ~zero.
        start = time.monotonic()
        with preflight.agent_slot(self.lock) as waited:
            self.assertLess(waited, 1.0)
        self.assertLess(time.monotonic() - start, 5.0)

    def test_agent_slot_waits_for_parallel_build(self):
        # A sibling session is mid-build (shared lock held): the agent slot
        # must block until it releases, not sail through and open claude.
        ctx = multiprocessing.get_context("spawn")
        started = ctx.Event()
        p = ctx.Process(target=_hold_build_lock, args=(str(self.lock), 2.0, started))
        p.start()
        try:
            self.assertTrue(started.wait(30), "builder never acquired the lock")
            with preflight.agent_slot(self.lock) as waited:
                pass
        finally:
            p.join(30)
        self.assertEqual(p.exitcode, 0)
        self.assertGreater(waited, 0.5)

    def test_agent_slots_run_concurrently(self):
        # Two agent runs hold the slot simultaneously (shared side) — the
        # barrier must not serialize AGENT runs against each other.
        ctx = multiprocessing.get_context("spawn")
        started, release = ctx.Event(), ctx.Event()
        p = ctx.Process(target=_hold_agent_slot, args=(str(self.lock), started, release))
        p.start()
        try:
            self.assertTrue(started.wait(30), "first agent never got its slot")
            start = time.monotonic()
            with preflight.agent_slot(self.lock) as waited:
                self.assertLess(waited, 1.0)
            self.assertLess(time.monotonic() - start, 5.0)
        finally:
            release.set()
            p.join(30)
        self.assertEqual(p.exitcode, 0)

    def test_staggered_setup_waits_for_open_agent_run(self):
        # THE STAGGERED-START RACE: an agent run is already open when a
        # sibling session begins its setup — `building()` must wait for the
        # run to finish instead of building under it.
        ctx = multiprocessing.get_context("spawn")
        started, release = ctx.Event(), ctx.Event()
        p = ctx.Process(target=_hold_agent_slot, args=(str(self.lock), started, release))
        p.start()
        try:
            self.assertTrue(started.wait(30), "agent never got its slot")
            entered_at = None
            start = time.monotonic()
            # Release the agent in 1.5s from a timer thread, so the
            # blocking building() below can be observed to wait then proceed.
            threading.Timer(1.5, release.set).start()
            with preflight.building(self.lock):
                entered_at = time.monotonic() - start
        finally:
            release.set()
            p.join(30)
        self.assertEqual(p.exitcode, 0)
        self.assertIsNotNone(entered_at)
        self.assertGreater(entered_at, 1.0, "setup did not wait for the open agent run")


if __name__ == "__main__":
    unittest.main()
