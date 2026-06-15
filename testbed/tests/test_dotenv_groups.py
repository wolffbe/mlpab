"""Unit tests for `.env` grouping in `_write_dotenv`.

`make setup` writes credentials grouped by platform/concern with a blank line
between groups (so e.g. the hopsworks block is visibly separated from the gcp
block), while `_load_dotenv` still round-trips every value (comments/blanks
are ignored on read).
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mlpab import cli


class DotenvGroupTests(unittest.TestCase):
    def _run(self, updates):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / ".env"
            with mock.patch.object(cli, "DOTENV_PATH", path):
                cli._write_dotenv(updates)
                text = path.read_text()
                # Round-trip: load it back into the environment.
                with mock.patch.dict(os.environ, {}, clear=False):
                    cli._load_dotenv()
            return text

    def test_groups_separated_by_blank_line_with_headers(self):
        text = self._run(
            {"HOPSWORKS_API_KEY": "h", "GCP_PROJECT": "g", "ANTHROPIC_API_KEY": "a"}
        )
        # Blank line separates groups.
        self.assertIn("\n\n", text)
        # Each group carries its header comment.
        self.assertIn("# Hopsworks", text)
        self.assertIn("# GCP", text)
        self.assertIn("# Agent — Claude", text)
        # Hopsworks and GCP are in distinct blocks (a blank line between them).
        blocks = text.strip().split("\n\n")
        hops_block = next(b for b in blocks if "HOPSWORKS_API_KEY" in b)
        gcp_block = next(b for b in blocks if "GCP_PROJECT" in b)
        self.assertIsNot(hops_block, gcp_block)

    def test_related_keys_cluster_in_one_block(self):
        text = self._run({"AZURE_TENANT_ID": "t", "AZUREML_WORKSPACE_NAME": "w"})
        blocks = text.strip().split("\n\n")
        azure_blocks = [b for b in blocks if "AZURE" in b]
        # Both AZURE_ and AZUREML_ land in the SAME Azure block.
        self.assertEqual(len(azure_blocks), 1)
        self.assertIn("AZURE_TENANT_ID", azure_blocks[0])
        self.assertIn("AZUREML_WORKSPACE_NAME", azure_blocks[0])

    def test_roundtrip_preserves_values(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / ".env"
            with mock.patch.object(cli, "DOTENV_PATH", path):
                cli._write_dotenv({"HOPSWORKS_HOST": "https://x", "GCP_PROJECT": "p"})
                with mock.patch.dict(os.environ, {}, clear=True):
                    cli._load_dotenv()
                    self.assertEqual(os.environ["HOPSWORKS_HOST"], "https://x")
                    self.assertEqual(os.environ["GCP_PROJECT"], "p")

    def test_unknown_key_goes_to_other(self):
        text = self._run({"SOME_RANDOM_KEY": "v"})
        self.assertIn("# Other", text)
        self.assertIn("SOME_RANDOM_KEY=v", text)


if __name__ == "__main__":
    unittest.main()
