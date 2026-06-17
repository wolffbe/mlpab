"""tmux session naming for parallel `mlpab start` (one config, many terminals).

Running `mlpab start <config>` in several terminals must each land in its own
tmux session — and therefore its own per-run Hopsworks project — instead of
colliding on the single `mlpab-<stem>` name.
"""

import unittest
from pathlib import Path
from unittest import mock

from mlpab import cli


class FreeSessionNameTests(unittest.TestCase):
    def test_bare_name_when_free(self):
        with mock.patch.object(cli, "_tmux_session_exists", return_value=False):
            self.assertEqual(cli._free_session_name(Path("configs/t.yaml")), "mlpab-t")

    def test_suffixes_when_base_taken(self):
        # base + -2 taken → next free is -3
        taken = {"mlpab-t", "mlpab-t-2"}
        with mock.patch.object(cli, "_tmux_session_exists", side_effect=lambda s: s in taken):
            self.assertEqual(cli._free_session_name(Path("configs/t.yaml")), "mlpab-t-3")

    def test_first_copy_is_suffix_2(self):
        taken = {"mlpab-t"}
        with mock.patch.object(cli, "_tmux_session_exists", side_effect=lambda s: s in taken):
            self.assertEqual(cli._free_session_name(Path("configs/t.yaml")), "mlpab-t-2")


class ConfigSessionsTests(unittest.TestCase):
    def test_collects_base_and_numbered_copies_only(self):
        listing = "mlpab-t\nmlpab-t-2\nmlpab-t-3\nmlpab-other\nmlpab-t-extra-7\n"
        completed = mock.Mock(returncode=0, stdout=listing)
        with mock.patch.object(cli.subprocess, "run", return_value=completed):
            got = cli._config_sessions(Path("configs/t.yaml"))
        # base + numbered copies; a different config (mlpab-other) is excluded.
        # `mlpab-t-extra-7` also starts with the prefix, so the family match is
        # by `mlpab-t` or `mlpab-t-*` — both numbered and any suffixed copy.
        self.assertEqual(got, ["mlpab-t", "mlpab-t-2", "mlpab-t-3", "mlpab-t-extra-7"])
        self.assertNotIn("mlpab-other", got)


if __name__ == "__main__":
    unittest.main()
