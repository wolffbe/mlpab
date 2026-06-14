"""Preflight orchestration after the cross-session build barrier was removed.

Parallel treatment sessions used to coordinate through a build barrier so a
fast session wouldn't open its timed agent while a sibling was still building.
That barrier is gone: every interface is built AND materialized into a
per-interface PREPARED venv once, up front, during the (serial) setup phase
(interfaces.prepare). Each run only CLONES its prepared venv read-only, so runs
do no build work and mutate no shared state — there's nothing left to serialize.

These tests lock in (a) that the barrier API is gone, and (b) that preflight
still verifies each unique interface exactly once.
"""

import unittest
from unittest import mock

from mlpab import interfaces, preflight


class BarrierRemovedTests(unittest.TestCase):
    def test_barrier_api_is_gone(self):
        # Regression guard: the barrier was deliberately removed in favour of
        # up-front prepared venvs. If a barrier is ever reintroduced, do it
        # consciously — not by resurrecting these names.
        for name in (
            "building",
            "agent_slot",
            "await_builds",
            "BUILD_BARRIER_LOCK",
            "AGENT_RUN_LOCK",
            "_run_lock_for",
        ):
            self.assertFalse(
                hasattr(preflight, name),
                f"preflight.{name} should no longer exist (barrier removed)",
            )


class PreflightDedupTests(unittest.TestCase):
    def test_each_unique_interface_verified_once(self):
        # preflight() de-duplicates requirements: interfaces.preflight (which
        # builds + materializes the prepared venv) runs once per unique
        # (platform, interface), no matter how many runs need it.
        seen = []

        def fake_iface_preflight(platform, interface, **kw):
            seen.append((platform, interface))
            return interfaces.InterfaceStatus(
                platform, interface, ok=True, installed=True, authenticated=True
            )

        reqs = [
            preflight.Requirement(platform="aws", interface="sdk"),
            preflight.Requirement(platform="aws", interface="sdk"),  # dup
            preflight.Requirement(platform="aws", interface="cli"),
        ]
        with mock.patch.object(interfaces, "preflight", side_effect=fake_iface_preflight):
            preflight.preflight(
                reqs,
                auth="api-key",
                model="claude-sonnet-4-6",
                probe_skills=False,
            )
        self.assertEqual(seen, [("aws", "sdk"), ("aws", "cli")])

    def test_first_failure_raises(self):
        def fake_iface_preflight(platform, interface, **kw):
            return interfaces.InterfaceStatus(
                platform, interface, ok=False, installed=False, authenticated=False, reason="boom"
            )

        with mock.patch.object(interfaces, "preflight", side_effect=fake_iface_preflight):
            with self.assertRaises(preflight.PreflightError):
                preflight.preflight(
                    [preflight.Requirement(platform="aws", interface="sdk")],
                    auth="api-key",
                    model="claude-sonnet-4-6",
                    probe_skills=False,
                )


if __name__ == "__main__":
    unittest.main()
