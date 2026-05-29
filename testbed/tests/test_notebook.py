"""Tests for the autoresearch analysis-notebook generator.

Covers the cell construction (deterministic, no kernel needed) and the
optional execution path. Execution-path tests skip if matplotlib + ipykernel
aren't both installed.
"""
import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from banter import notebook as notebook_mod


def _has_exec_deps() -> bool:
    try:
        import matplotlib  # noqa: F401
        import ipykernel  # noqa: F401
        from jupyter_client.manager import KernelManager  # noqa: F401
        return True
    except ImportError:
        return False


def _write_results_csv(parent: Path, rows: list[dict]) -> Path:
    """Drop a minimal master results.csv into `parent`. Field set is derived
    from the union of all keys in `rows` so each test only declares the
    columns it cares about."""
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    path = parent / "results.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


class VersionIndexTests(unittest.TestCase):
    def test_extracts_integer(self):
        self.assertEqual(notebook_mod._version_index("v0"), 0)
        self.assertEqual(notebook_mod._version_index("v12"), 12)

    def test_returns_none_for_non_incr(self):
        self.assertIsNone(notebook_mod._version_index("interface"))
        self.assertIsNone(notebook_mod._version_index("vx"))


class BuildRunNotebookTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        # Layout mirrors production: master at `<root>/results.csv`,
        # per-session run dir at `<root>/<session>/analysis.ipynb`.
        self.run_dir = self.tmp / "session"
        self.run_dir.mkdir()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_returns_none_when_no_results_csv(self):
        self.assertIsNone(notebook_mod.build_run_notebook(self.run_dir, [("score", "maximize")], "0"))

    def test_returns_none_when_csv_has_no_increments_for_session(self):
        # Empty master + a session id that doesn't appear → no notebook.
        _write_results_csv(self.run_dir, [])
        self.assertIsNone(notebook_mod.build_run_notebook(self.run_dir, [("score", "maximize")], "0"))

    def test_charts_all_tracked_metrics_regardless_of_goals(self):
        # Every metric in results.TRACKED_METRICS is always charted; goals
        # only mark which are flagged "(optimized)" in titles. The notebook
        # reads raw per-challenge rows from the master CSV and computes
        # per-increment means in-cell.
        from banter import results as results_mod
        _write_results_csv(self.run_dir, [
            {"run": "7", "version": "v0", "challenge": "c1", "score": "0.5",
             "total_tokens": "100", "wall_time_s": "10", "cli_calls": "0", "python_calls": "5"},
            {"run": "7", "version": "v1", "challenge": "c1", "score": "0.7",
             "total_tokens": "120", "wall_time_s": "12", "cli_calls": "2", "python_calls": "3"},
            # Noise from another session — must be filtered out.
            {"run": "9", "version": "v0", "challenge": "c1", "score": "0.1",
             "total_tokens": "9", "wall_time_s": "1", "cli_calls": "0", "python_calls": "9"},
        ])
        goals = [("score", "maximize")]
        nb_path = notebook_mod.build_run_notebook(self.run_dir, goals, "7", execute=False)
        self.assertIsNotNone(nb_path)
        nb = json.loads(nb_path.read_text())

        cells = nb["cells"]
        # Layout: [intro, "## Setup", setup_code, "## Metrics …", then
        # 2 cells (header + chart) per tracked metric.
        self.assertEqual(len(cells), 4 + 2 * len(results_mod.TRACKED_METRICS))

        intro_src = "".join(cells[0]["source"]) if isinstance(cells[0]["source"], list) else cells[0]["source"]
        self.assertIn("Autoresearch run `7`", intro_src)
        # Intro lists every tracked metric; only `score` is marked optimized.
        for metric, direction in results_mod.TRACKED_METRICS:
            self.assertIn(f"**{direction}**(`{metric}`)", intro_src)
        self.assertIn("**maximize**(`score`) — **optimized**", intro_src)

        for i, (metric, direction) in enumerate(results_mod.TRACKED_METRICS):
            hdr = cells[4 + 2 * i]
            code = cells[4 + 2 * i + 1]
            hdr_src = "".join(hdr["source"]) if isinstance(hdr["source"], list) else hdr["source"]
            code_src = "".join(code["source"]) if isinstance(code["source"], list) else code["source"]
            self.assertEqual(hdr["cell_type"], "markdown")
            self.assertIn(f"`{direction}({metric})`", hdr_src)
            if metric == "score":
                self.assertIn("optimized", hdr_src)
            self.assertEqual(code["cell_type"], "code")
            self.assertIn(json.dumps(metric), code_src)
            self.assertIn(json.dumps(direction), code_src)

    def test_extra_config_goals_chart_after_tracked_set(self):
        # A goal whose metric is NOT in TRACKED_METRICS still gets charted —
        # appended after the canonical list.
        _write_results_csv(self.run_dir, [
            {"run": "1", "version": "v0", "challenge": "c1", "score": "0.5",
             "total_tokens": "100", "wall_time_s": "10", "cli_calls": "0", "python_calls": "5"},
        ])
        # `_write_results_csv` only carries TRACKED_METRICS columns; the
        # "cost_usd" column ends up missing → notebook code prints a skip.
        goals = [("cost_usd", "minimize")]
        nb_path = notebook_mod.build_run_notebook(self.run_dir, goals, "1", execute=False)
        nb = json.loads(nb_path.read_text())
        # Find the markdown header for the extra metric.
        headers = [
            ("".join(c["source"]) if isinstance(c["source"], list) else c["source"])
            for c in nb["cells"] if c["cell_type"] == "markdown"
        ]
        self.assertTrue(any("`minimize(cost_usd)`" in h for h in headers),
                        f"extra goal not charted; got headers: {headers}")

    @unittest.skipUnless(_has_exec_deps(), "needs matplotlib + ipykernel")
    def test_executed_notebook_embeds_plot_images(self):
        # Two increments × two challenges each → the notebook groups by
        # increment and averages. Confirms the kernel runs and embeds PNGs.
        _write_results_csv(self.run_dir, [
            {"run": "9", "version": "v0", "challenge": "c1", "score": "0.5",
             "total_tokens": "100", "wall_time_s": "10", "cli_calls": "0", "python_calls": "5"},
            {"run": "9", "version": "v0", "challenge": "c2", "score": "0.6",
             "total_tokens": "110", "wall_time_s": "11", "cli_calls": "1", "python_calls": "4"},
            {"run": "9", "version": "v1", "challenge": "c1", "score": "0.8",
             "total_tokens": "120", "wall_time_s": "12", "cli_calls": "2", "python_calls": "3"},
            {"run": "9", "version": "v1", "challenge": "c2", "score": "0.85",
             "total_tokens": "130", "wall_time_s": "13", "cli_calls": "3", "python_calls": "2"},
        ])
        goals = [("score", "maximize")]
        nb_path = notebook_mod.build_run_notebook(self.run_dir, goals, "9", execute=True)
        self.assertIsNotNone(nb_path)
        nb = json.loads(nb_path.read_text())
        # The last cell is the final chart cell (any TRACKED_METRICS chart).
        chart_cell = nb["cells"][-1]
        self.assertEqual(chart_cell["cell_type"], "code")
        outs = chart_cell.get("outputs", [])
        has_png = any("image/png" in (o.get("data") or {}) for o in outs)
        self.assertTrue(has_png, f"no image/png output in chart cell: {outs}")


if __name__ == "__main__":
    unittest.main()
