"""Unit tests for `mlpab setup` Claude auth: minting the long-lived OAuth token.

`_mint_setup_token` runs `claude setup-token` and extracts the sk-ant-oat01-…
token from stdout, falling back to a paste prompt. This is the credential that
must be pinned in .env so overnight headless runs don't fall back to the
short-lived Keychain token (which expires mid-run).
"""

import subprocess
import unittest
from unittest import mock

from mlpab import cli


class MintSetupTokenTests(unittest.TestCase):
    def test_extracts_token_from_stdout(self):
        out = "Visit https://… to authorize.\nYour token:\nsk-ant-oat01-AbC_123-def\n"
        with mock.patch("subprocess.run", return_value=mock.Mock(stdout=out)) as run:
            tok = cli._mint_setup_token()
        self.assertEqual(tok, "sk-ant-oat01-AbC_123-def")
        # Captures stdout so it can be scanned (stderr/prompts stay on the terminal).
        self.assertEqual(run.call_args.args[0], ["claude", "setup-token"])
        self.assertEqual(run.call_args.kwargs["stdout"], subprocess.PIPE)

    def test_falls_back_to_paste_when_stdout_has_no_token(self):
        with (
            mock.patch("subprocess.run", return_value=mock.Mock(stdout="no token here\n")),
            mock.patch("click.prompt", return_value="sk-ant-oat01-PASTED_xyz"),
        ):
            tok = cli._mint_setup_token()
        self.assertEqual(tok, "sk-ant-oat01-PASTED_xyz")

    def test_extracts_token_embedded_in_pasted_text(self):
        with (
            mock.patch("subprocess.run", return_value=mock.Mock(stdout="")),
            mock.patch("click.prompt", return_value="  token=sk-ant-oat01-EMB next  "),
        ):
            tok = cli._mint_setup_token()
        self.assertEqual(tok, "sk-ant-oat01-EMB")

    def test_returns_none_when_nothing_provided(self):
        with (
            mock.patch("subprocess.run", return_value=mock.Mock(stdout="")),
            mock.patch("click.prompt", return_value=""),
        ):
            self.assertIsNone(cli._mint_setup_token())


if __name__ == "__main__":
    unittest.main()
