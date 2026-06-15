"""Unit tests for the Hopsworks platform teardown (configs/platforms/hopsworks/
teardown.py), focused on per-run SCOPING: with HOPSWORKS_PROJECT set, only the
run's own project is deleted (so parallel runs don't tear down each other's
work); unset, the whole-key sweep deletes every owned project. Offline — a fake
hopsworks SDK + REST client, no cluster."""

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

_TEARDOWN = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "platforms"
    / "hopsworks"
    / "teardown.py"
)

# uid of the API key's user; a second uid owns a project we must never touch.
_MY_UID = 1
_OTHER_UID = 2


class _FakeRestClient:
    """Answers the two requests teardown makes: list projects, delete a project.
    Records deleted project ids."""

    def __init__(self, teams):
        self._teams = teams
        self.deleted: list[int] = []

    def _send_request(self, method, path):
        if method == "GET" and path == ["project"]:
            return self._teams
        if method == "POST" and path[0] == "project" and path[-1] == "delete":
            self.deleted.append(path[1])
            return {}
        raise AssertionError(f"unexpected request {method} {path}")


def _team(pid, name, owner_uid):
    return {
        "user": {"uid": _MY_UID},
        "project": {"id": pid, "name": name, "owner": {"uid": owner_uid}},
    }


def _run_main(rest_client):
    """Run teardown.main() entirely offline: teardown imports `hopsworks` +
    `hopsworks_common.client` INSIDE main(), so the fakes must be live while it
    runs (not merely while the module loads)."""
    fake_hopsworks = mock.MagicMock()
    fake_hopsworks.login.return_value = None
    fake_client_mod = mock.MagicMock()
    fake_client_mod.get_instance.return_value = rest_client
    fake_common = mock.MagicMock()
    fake_common.client = fake_client_mod
    mods = {
        "hopsworks": fake_hopsworks,
        "hopsworks_common": fake_common,
        "hopsworks_common.client": fake_client_mod,
    }
    spec = importlib.util.spec_from_file_location("hopsworks_teardown", _TEARDOWN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with mock.patch.dict(sys.modules, mods):
        mod.main()


class TeardownScopeTests(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.pop("HOPSWORKS_PROJECT", None)
        # Two projects owned by us (ids 10, 20) + one owned by someone else (30).
        self.teams = [
            _team(10, "mlpabAAA", _MY_UID),
            _team(20, "mlpabBBB", _MY_UID),
            _team(30, "someone-else", _OTHER_UID),
        ]

    def tearDown(self):
        os.environ.pop("HOPSWORKS_PROJECT", None)
        if self._saved is not None:
            os.environ["HOPSWORKS_PROJECT"] = self._saved

    def test_scoped_deletes_only_the_runs_project(self):
        os.environ["HOPSWORKS_PROJECT"] = "mlpabAAA"
        rest = _FakeRestClient(self.teams)
        _run_main(rest)
        # Only our run's project — not the other owned project, not the foreign one.
        self.assertEqual(rest.deleted, [10])

    def test_scoped_missing_project_deletes_nothing(self):
        # The run's project already gone (e.g. start-teardown before setup).
        os.environ["HOPSWORKS_PROJECT"] = "mlpabZZZ"
        rest = _FakeRestClient(self.teams)
        _run_main(rest)
        self.assertEqual(rest.deleted, [])

    def test_unset_sweeps_all_owned_projects(self):
        rest = _FakeRestClient(self.teams)
        _run_main(rest)
        # Whole-key fallback: both owned projects, never the foreign one.
        self.assertEqual(sorted(rest.deleted), [10, 20])


if __name__ == "__main__":
    unittest.main()
