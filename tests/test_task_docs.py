"""Unit tests for the mechanical task-docs exporter (evals.export_docs)."""

import importlib
import tempfile
import unittest
from pathlib import Path

from evals import export_docs


class DiscoveryAndConstantsTests(unittest.TestCase):
    def test_discovers_every_category(self):
        tasks = export_docs.discover()
        self.assertEqual({c for c, _ in tasks}, set(export_docs.CATEGORIES))
        self.assertGreaterEqual(len(tasks), 26)

    def test_every_package_declares_kind_summary_docstring(self):
        for cat, name in export_docs.discover():
            mod = importlib.import_module(f"evals.{cat}.{name}.generate")
            self.assertIn(mod.KIND, export_docs.KINDS, f"{cat}/{name}: bad KIND")
            self.assertTrue(mod.SUMMARY.strip(), f"{cat}/{name}: empty SUMMARY")
            self.assertTrue((mod.__doc__ or "").strip(), f"{cat}/{name}: no docstring")


class RenderHelpersTests(unittest.TestCase):
    def test_strip_usage_drops_the_block_and_keeps_prose(self):
        doc = (
            "Title line.\n\nUsage:\n    python -m x --seed 1\n"
            "    python -m x --selftest\n\nBody prose."
        )
        stripped = export_docs.strip_usage(doc)
        self.assertIn("Title line.", stripped)
        self.assertIn("Body prose.", stripped)
        self.assertNotIn("--selftest", stripped)

    def test_grader_shape_covers_the_three_clis(self):
        self.assertEqual(export_docs.grader_shape("… grade_answers_main(…)"), "answers")
        self.assertEqual(export_docs.grader_shape("… grade_platform_main(…)"), "platform")
        self.assertEqual(
            export_docs.grader_shape("from evals.capstone.common import grade_main"), "platform"
        )
        self.assertEqual(export_docs.grader_shape("custom --csv grader (pit)"), "table")


class ExportEndToEndTests(unittest.TestCase):
    def test_export_ingest_page(self):
        with tempfile.TemporaryDirectory() as t:
            page = export_docs.export_task("feature", "ingest", Path(t))
            text = page.read_text()
        for expected in (
            export_docs.MARKER,
            "# ingest (feature)",
            "## Prompt (seed 1)",
            "transactions000001",  # seed-1 instance suffix in the literal prompt
            "## Staged files",
            "`data/schema.md`",
            "## Assert suite",
            "`A3_content`",
            "keep_overlap",
        ):
            self.assertIn(expected, text)


if __name__ == "__main__":
    unittest.main()
