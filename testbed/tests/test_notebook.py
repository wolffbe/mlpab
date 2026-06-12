"""Tests for the global results-notebook generator.

Covers cell construction (deterministic, no kernel needed) and the optional
execution path. Execution-path tests skip if matplotlib + ipykernel aren't both
installed.
"""
import csv
import shutil
import tempfile
import unittest
from pathlib import Path

import nbformat

from banter import notebook as notebook_mod
from banter import results as results_mod


def _has_exec_deps() -> bool:
    try:
        import matplotlib  # noqa: F401
        import ipykernel  # noqa: F401
        from jupyter_client.manager import KernelManager  # noqa: F401
        return True
    except ImportError:
        return False


def _write_results_csv(parent: Path, rows: list[dict]) -> Path:
    path = parent / "results.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results_mod.RESULTS_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in results_mod.RESULTS_FIELDS})
    return path


def _bm_row(config, task, asserts_passed, n=1, interface="cli",
            asserts_total=4, wall_time_s=10, total_tokens=100, cost_usd=0.1):
    return {
        "config": config, "model": "claude-opus-4-8", "platform": "hopsworks",
        "interface": interface, "version": "v1", "skills": "none",
        "task": task, "n": str(n),
        "asserts_passed": str(asserts_passed), "asserts_total": str(asserts_total),
        "wall_time_s": str(wall_time_s), "total_tokens": str(total_tokens),
        "cost_usd": str(cost_usd),
    }


class BuildResultsNotebookTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.results_root = self.tmp / "results"
        self.results_root.mkdir()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_returns_none_when_no_results_csv(self):
        self.assertIsNone(notebook_mod.build_results_notebook(self.results_root))

    def test_returns_none_when_csv_empty(self):
        _write_results_csv(self.results_root, [])
        self.assertIsNone(notebook_mod.build_results_notebook(self.results_root))

    def test_sections_per_config_and_chart_per_tracked_metric(self):
        _write_results_csv(self.results_root, [
            _bm_row("rq1-a", "drift", 2, n=1),
            _bm_row("rq1-a", "drift", 3, n=2),
            _bm_row("rq1-b", "skew", 4, interface="sdk"),
        ])
        nb_path = notebook_mod.build_results_notebook(self.results_root, execute=False)
        self.assertIsNotNone(nb_path)
        self.assertEqual(nb_path.name, "results.ipynb")
        nb = nbformat.read(nb_path, as_version=4)
        code = [c for c in nb.cells if c.cell_type == "code"]
        md = [c.source for c in nb.cells if c.cell_type == "markdown"]
        n_metrics = len(results_mod.RESULTS_TRACKED_METRICS)
        # Setup + aggregate + (2 configs × 1 model × metrics) chart cells.
        self.assertEqual(len(code), 2 + 2 * n_metrics)
        # The setup cell loads results.csv; the aggregate cell averages over
        # the full (config … skills) identity.
        self.assertTrue(any("read_csv" in c.source for c in code))
        self.assertTrue(any("groupby(GROUP" in c.source for c in code))
        # Config headline / model sub-headline structure.
        self.assertIn("## Config `rq1-a`", md)
        self.assertIn("## Config `rq1-b`", md)
        self.assertIn("### Model `claude-opus-4-8`", md)
        # Charts bar per (platform, interface, version, skills) + average line.
        chart = next(c.source for c in code if "axhline" in c.source)
        self.assertIn('"platform", "interface", "version", "skills"', chart)

    @unittest.skipUnless(_has_exec_deps(), "needs matplotlib + ipykernel")
    def test_executed_notebook_embeds_outputs(self):
        _write_results_csv(self.results_root, [
            _bm_row("rq1-a", "drift", 2, n=1),
            _bm_row("rq1-a", "drift", 3, n=2),
        ])
        nb_path = notebook_mod.build_results_notebook(self.results_root, execute=True)
        nb = nbformat.read(nb_path, as_version=4)
        self.assertTrue(any(c.cell_type == "code" and c.get("outputs") for c in nb.cells))


if __name__ == "__main__":
    unittest.main()
