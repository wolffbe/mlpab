"""Tests for the benchmark analysis-notebook generator.

Covers cell construction (deterministic, no kernel needed) and the optional
execution path. Execution-path tests skip if matplotlib + ipykernel aren't both
installed. (Autoresearch's global notebook is covered in test_experiments.py.)
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


def _write_benchmark_csv(parent: Path, rows: list[dict]) -> Path:
    path = parent / "results.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results_mod.BENCHMARK_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in results_mod.BENCHMARK_FIELDS})
    return path


def _bm_row(task, challenge, score, wall_time_s=10, total_tokens=100, cost_usd=0.1):
    return {
        "platform": "none", "interface": "none", "skills": "none",
        "task": task, "challenge": challenge, "valid_submission": "1",
        "score": str(score), "wall_time_s": str(wall_time_s),
        "total_tokens": str(total_tokens), "cost_usd": str(cost_usd),
    }


class BuildBenchmarkNotebookTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.run_dir = self.tmp / "0_none"
        self.run_dir.mkdir()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_returns_none_when_no_results_csv(self):
        self.assertIsNone(notebook_mod.build_benchmark_notebook(self.run_dir, "0"))

    def test_returns_none_when_csv_empty(self):
        _write_benchmark_csv(self.run_dir, [])
        self.assertIsNone(notebook_mod.build_benchmark_notebook(self.run_dir, "0"))

    def test_writes_notebook_with_a_chart_per_tracked_metric(self):
        _write_benchmark_csv(self.run_dir, [
            _bm_row("tabular", "a", 0.5),
            _bm_row("image", "b", 0.9),
        ])
        nb_path = notebook_mod.build_benchmark_notebook(self.run_dir, "0", execute=False)
        self.assertIsNotNone(nb_path)
        nb = nbformat.read(nb_path, as_version=4)
        # One markdown + one code cell per tracked metric, plus setup/intro cells.
        code = [c for c in nb.cells if c.cell_type == "code"]
        self.assertGreaterEqual(len(code), len(results_mod.BENCHMARK_TRACKED_METRICS))
        # The setup cell loads results.csv.
        self.assertTrue(any("read_csv" in c.source for c in code))

    @unittest.skipUnless(_has_exec_deps(), "needs matplotlib + ipykernel")
    def test_executed_notebook_embeds_outputs(self):
        _write_benchmark_csv(self.run_dir, [_bm_row("tabular", "a", 0.5)])
        nb_path = notebook_mod.build_benchmark_notebook(self.run_dir, "0", execute=True)
        nb = nbformat.read(nb_path, as_version=4)
        self.assertTrue(any(c.cell_type == "code" and c.get("outputs") for c in nb.cells))


if __name__ == "__main__":
    unittest.main()
