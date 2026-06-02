"""Tests for the experiment results registry + composite math."""
import csv
import tempfile
import unittest
from pathlib import Path

from banter import autoresearch as ar
from banter import experiments


def _make_cfg(root, rel, goals, *, platform="hopsworks", interface="cli",
              max_seconds=14400, improve=("interface",)):
    """An in-memory AutoresearchConfig (as load_config would produce) carrying the
    config_path that identifies its rows in the global table."""
    return ar.AutoresearchConfig(
        tasks={"tabular": ["a", "b", "c", "d"]}, challenges=["a", "b", "c", "d"],
        interfaces=[ar.InterfaceRef(platform=platform, interface=interface)],
        skills="none", docs="none",
        goals=[ar.Goal(m, d) for m, d in goals],
        budget=ar.Budget(max_seconds=max_seconds), improve=list(improve),
        engineer_model="claude-sonnet-4-5", researcher_model="claude-opus-4-7",
        experiment=None, treatment=None,
        config_path=str(root / rel),
    )


def _run_row(version, task, challenge, *, score, python_calls=0, total_tokens=1000,
             cli=1, run_dir="rd"):
    """A results.Row as a dict, as the runner passes it to append_run."""
    return {
        "run": "exp", "version": version, "platform": "hopsworks", "interface": "cli",
        "skills": "none", "task": task, "challenge": challenge,
        "valid_submission": "1", "score": str(score), "medal": "",
        "total_tokens": str(total_tokens), "python_calls": str(python_calls),
        "cli_calls": str(cli), "run_dir": run_dir,
    }


class CompositeMathTests(unittest.TestCase):
    PER_VERSION = {
        0: {"python_calls": 0, "score": 0.500, "total_tokens": 4914},
        1: {"python_calls": 0, "score": 0.966, "total_tokens": 4844},
        2: {"python_calls": 0, "score": 0.960, "total_tokens": 4562},
        3: {"python_calls": 0, "score": 0.983, "total_tokens": 4613},
    }
    MULTI = [("python_calls", "minimize"), ("score", "maximize"), ("total_tokens", "minimize")]

    def test_aggregate_means_across_challenges(self):
        rows = [
            {"version": "v0", "score": "0.4", "total_tokens": "100"},
            {"version": "v0", "score": "0.6", "total_tokens": "200"},
            {"version": "v1", "score": "0.9", "total_tokens": "300"},
        ]
        agg = experiments.aggregate_by_version(rows, ["score", "total_tokens"])
        self.assertAlmostEqual(agg[0]["score"], 0.5)
        self.assertAlmostEqual(agg[0]["total_tokens"], 150.0)

    def test_multivariate_composite_picks_v2(self):
        self.assertEqual(experiments.best_version(self.PER_VERSION, self.MULTI), 2)

    def test_breakdown_exposes_per_goal_and_flat_goal_zero(self):
        bd = experiments.composite_breakdown(self.PER_VERSION, self.MULTI)
        # python_calls is flat → contributes 0 to every version.
        self.assertEqual(bd[0]["contributions"]["python_calls"], 0.0)
        self.assertEqual(bd[2]["contributions"]["python_calls"], 0.0)
        # v2 is best on tokens (min) → full contribution there.
        self.assertEqual(bd[2]["contributions"]["total_tokens"], 1.0)
        # composite = sum of contributions.
        self.assertAlmostEqual(bd[2]["composite"], sum(bd[2]["contributions"].values()))

    def test_flat_composite_ties_to_baseline(self):
        # No goal has spread (identical across versions) → tie → LOWEST version
        # (the honest "best" is the baseline, not a later no-op version).
        per = {0: {"python_calls": 2, "score": 0.9}, 1: {"python_calls": 2, "score": 0.9}}
        self.assertEqual(experiments.best_version(per, [("python_calls", "minimize"),
                                                        ("score", "maximize")]), 0)
        # Single baseline-only version → itself.
        self.assertEqual(experiments.best_version({0: {"score": 0.5}}, [("score", "maximize")]), 0)

    def test_missing_goal_value_counts_as_worst_not_skipped(self):
        # v1 is missing `score` (e.g. a failed challenge) → it must NOT get a
        # free pass: its composite sums the SAME goals as v0 (score contributes 0).
        per = {
            0: {"score": 0.9, "python_calls": 5},
            1: {"score": None, "python_calls": 1},
        }
        bd = experiments.composite_breakdown(per, [("score", "maximize"), ("python_calls", "minimize")])
        self.assertEqual(bd[1]["contributions"]["score"], 0.0)   # missing → worst, not skipped
        self.assertIn("score", bd[0]["contributions"])
        # Both versions' composites sum the same number of goal terms.
        self.assertEqual(len(bd[0]["contributions"]), len(bd[1]["contributions"]))

    def test_univariate_picks_min_python(self):
        per = {0: {"python_calls": 5}, 1: {"python_calls": 2}, 2: {"python_calls": 8}}
        self.assertEqual(experiments.best_version(per, [("python_calls", "minimize")]), 1)

    def test_goals_for_interface_filters_delegation(self):
        goals = [("score", "maximize"), ("python_calls", "minimize"),
                 ("cli_calls", "maximize"), ("mcp_calls", "maximize"), ("sdk_calls", "maximize")]
        # mcp leaf → only mcp_calls survives among the delegation goals.
        self.assertEqual(experiments.goals_for_interface(goals, "mcp"),
                         [("score", "maximize"), ("python_calls", "minimize"), ("mcp_calls", "maximize")])
        self.assertEqual(experiments.goals_for_interface(goals, "cli"),
                         [("score", "maximize"), ("python_calls", "minimize"), ("cli_calls", "maximize")])
        # No delegation goals → unchanged.
        plain = [("python_calls", "minimize"), ("score", "maximize")]
        self.assertEqual(experiments.goals_for_interface(plain, "sdk"), plain)

    def test_goals_str_roundtrip(self):
        import json
        s = experiments._goals_str([("python_calls", "minimize"), ("score", "maximize")])
        self.assertEqual(json.loads(s), ["python_calls:min", "score:max"])   # JSON array
        self.assertEqual(experiments.parse_goals_str(s),
                         [("python_calls", "minimize"), ("score", "maximize")])
        # Legacy pipe form still parses.
        self.assertEqual(experiments.parse_goals_str("python_calls:min|score:max"),
                         [("python_calls", "minimize"), ("score", "maximize")])


class AppendRunTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.results_root = self.root / "results"
        # A treatment config carries config_path + goals + budget (univariate).
        self.cfg = _make_cfg(
            self.root, "platforms/hopsworks/autoresearch/rq1/t01-cli-univariate.yaml",
            [("python_calls", "minimize")])

    def _rows(self):
        with experiments.table_path(self.results_root).open(newline="") as f:
            return list(csv.DictReader(f))

    def test_append_writes_exploded_rows_with_design(self):
        experiments.append_run(self.results_root, self.cfg, _run_row("v0", "tabular", "a", score=0.5, python_calls=3))
        experiments.append_run(self.results_root, self.cfg, _run_row("v1", "tabular", "a", score=0.7, python_calls=1))
        rows = self._rows()
        self.assertEqual(len(rows), 2)
        v1 = [r for r in rows if r["version"] == "v1"][0]
        self.assertEqual(v1["score"], "0.7")
        self.assertEqual(v1["python_calls"], "1")
        self.assertEqual(v1["challenge"], "a")
        # Design metadata from the config, repeated per row.
        import json
        self.assertEqual(v1["config"], "platforms/hopsworks/autoresearch/rq1/t01-cli-univariate.yaml")
        self.assertEqual(json.loads(v1["goals"]), ["python_calls:min"])         # JSON array
        self.assertEqual(v1["optimization_variable"], "univariate")             # derived: 1 goal
        # budget is a JSON array of specified budgets (iterations default 11 + max_seconds).
        self.assertEqual(json.loads(v1["budget"]), ["iterations:11", f"max_seconds:{1 * 4 * 3600}"])

    def test_started_at_and_eng_columns_written(self):
        row = _run_row("v0", "tabular", "a", score=0.5)
        row["started_at"] = "2026-06-02T12:00:00+00:00"
        row["eng_wall_time_s"] = "31.96"
        row["eng_cost_usd"] = "0.29"
        row["total_wall_time_s"] = "31.96"
        row["total_cost"] = "0.29"
        experiments.append_run(self.results_root, self.cfg, row)
        r = self._rows()[0]
        self.assertEqual(r["started_at"], "2026-06-02T12:00:00+00:00")
        self.assertEqual(r["eng_wall_time_s"], "31.96")
        self.assertEqual(r["eng_cost_usd"], "0.29")
        self.assertEqual(r["res_wall_time_s"], "")          # not yet attributed

    def test_attribute_researcher_splits_and_totals(self):
        for ch in ("a", "b"):
            row = _run_row("v0", "tabular", ch, score=0.5)
            row["eng_input_tokens"] = "100"
            row["eng_output_tokens"] = "50"
            row["eng_total_tokens"] = "150"
            row["eng_wall_time_s"] = "10"
            row["eng_cost_usd"] = "1.0"
            row["total_tokens"] = "150"
            row["total_wall_time_s"] = "10"
            row["total_cost"] = "1.0"
            experiments.append_run(self.results_root, self.cfg, row)
        # Researcher usage split across the 2 rows.
        experiments.attribute_researcher(self.results_root, self.cfg, "hopsworks", "cli", {
            "input_tokens": 200, "output_tokens": 100, "total_tokens": 300,
            "wall_s": 20.0, "cost_usd": 4.0,
        })
        r = self._rows()[0]
        # researcher per-row share (÷2)
        self.assertEqual(float(r["res_input_tokens"]), 100.0)
        self.assertEqual(float(r["res_total_tokens"]), 150.0)
        self.assertEqual(float(r["res_wall_time_s"]), 10.0)
        self.assertEqual(float(r["res_cost_usd"]), 2.0)
        # total = engineer + researcher
        self.assertEqual(float(r["total_input_tokens"]), 200.0)   # 100 + 100
        self.assertEqual(float(r["total_output_tokens"]), 100.0)  # 50 + 50
        self.assertEqual(float(r["total_tokens"]), 300.0)         # 150 + 150
        self.assertEqual(float(r["total_wall_time_s"]), 20.0)     # 10 + 10
        self.assertEqual(float(r["total_cost"]), 3.0)             # 1 + 2

    def test_append_filters_goals_to_row_interface(self):
        # A config maximizing all delegation metrics; the mcp row should record
        # only mcp_calls among them.
        cfg = ar.AutoresearchConfig(
            tasks={"t": ["a"]}, challenges=["a"],
            interfaces=[ar.InterfaceRef(platform="mlkit", interface="mcp")],
            skills="none", docs="none",
            goals=[ar.Goal("score", "maximize"), ar.Goal("cli_calls", "maximize"),
                   ar.Goal("mcp_calls", "maximize"), ar.Goal("sdk_calls", "maximize")],
            budget=ar.Budget(), improve=["interface"],
            engineer_model="m", researcher_model="r",
            experiment=None, treatment=None,
            config_path=str(self.root / "platforms/mlkit/autoresearch/mlkit-test-ar.yaml"),
        )
        row = _run_row("v0", "t", "a", score=0.5)
        row["platform"] = "mlkit"
        row["interface"] = "mcp"
        experiments.append_run(self.results_root, cfg, row)
        import json
        r = self._rows()[0]
        self.assertEqual(json.loads(r["goals"]), ["score:max", "mcp_calls:max"])
        # 2 surviving goals → bivariate (derived from the filtered goal count).
        self.assertEqual(r["optimization_variable"], "bivariate")

    def test_annotate_version_fills_all_challenge_rows(self):
        for ch in ("a", "b"):
            experiments.append_run(self.results_root, self.cfg, _run_row("v0", "tabular", ch, score=0.5))
        cfg_rel = experiments._config_rel(self.cfg, self.results_root)
        n = experiments.annotate_version(self.results_root, cfg_rel, "v0", {
            "hypothesis": "H", "change": "baseline", "verdict": "neutral", "keep": "1",
            "observations": "obs", "proposed_changes": "next",
        })
        self.assertEqual(n, 2)   # both challenge rows of v0
        rows = self._rows()
        self.assertTrue(all(r["hypothesis"] == "H" and r["verdict"] == "neutral"
                            and r["keep"] == "1" for r in rows))
        # v1 (not yet annotated) untouched.
        experiments.append_run(self.results_root, self.cfg, _run_row("v1", "tabular", "a", score=0.7))
        self.assertEqual(
            experiments.annotate_version(self.results_root, cfg_rel, "v1", {"verdict": "positive"}), 1)

    def test_clear_treatment_removes_only_that_config(self):
        experiments.append_run(self.results_root, self.cfg, _run_row("v0", "tabular", "a", score=0.5))
        # A second treatment's row.
        cfg2 = _make_cfg(
            self.root, "platforms/hopsworks/autoresearch/rq1/t02-cli-bivariate.yaml",
            [("python_calls", "minimize"), ("score", "maximize")])
        experiments.append_run(self.results_root, cfg2, _run_row("v0", "tabular", "a", score=0.6))
        experiments.clear_treatment(self.results_root, self.cfg)
        rows = self._rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["config"], "platforms/hopsworks/autoresearch/rq1/t02-cli-bivariate.yaml")


class ResearchMcpTests(unittest.TestCase):
    def setUp(self):
        from banter import research_mcp
        self.mcp = research_mcp
        self.root = Path(tempfile.mkdtemp())
        self.results_root = self.root / "results"
        cfg = _make_cfg(
            self.root, "platforms/hopsworks/autoresearch/rq3/t21-cli-multivariate.yaml",
            [("python_calls", "minimize"), ("score", "maximize"), ("total_tokens", "minimize")])
        for r in [
            _run_row("v0", "tab", "x", score=0.5, total_tokens=4914),
            _run_row("v1", "tab", "x", score=0.96, total_tokens=4844),
            _run_row("v2", "tab", "x", score=0.96, total_tokens=4562),
        ]:
            experiments.append_run(self.results_root, cfg, r)
        self._cfg = cfg

    def _env(self):
        import os
        os.environ["BANTER_EXPERIMENTS_CSV"] = str(experiments.table_path(self.results_root))
        os.environ["BANTER_CONFIG"] = experiments._config_rel(self._cfg, self.results_root)
        os.environ["BANTER_PLATFORM"] = "hopsworks"
        os.environ["BANTER_INTERFACE"] = "cli"
        os.environ["BANTER_GOALS"] = "python_calls:min|score:max|total_tokens:min"

    def test_initialize_and_tools_list(self):
        init = self.mcp._handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        self.assertEqual(init["result"]["serverInfo"]["name"], "banter-research")
        listed = self.mcp._handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        names = [t["name"] for t in listed["result"]["tools"]]
        self.assertEqual(names, ["normalized_composite"])

    def test_tool_call_returns_all_versions_with_breakdown(self):
        self._env()
        resp = self.mcp._handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                                 "params": {"name": "normalized_composite", "arguments": {}}})
        import json
        payload = json.loads(resp["result"]["content"][0]["text"])
        # v2: best tokens + tied-best score → composite winner over v1.
        self.assertEqual(payload["best_version"], "v2")
        # All versions (full history), not just the best.
        self.assertEqual(sorted(payload["versions"]), ["v0", "v1", "v2"])
        v2 = payload["versions"]["v2"]
        # Composite + per-goal breakdown (value, direction, normalized contribution).
        self.assertIn("composite", v2)
        self.assertEqual(set(v2["goals"]), {"python_calls", "score", "total_tokens"})
        self.assertEqual(v2["goals"]["score"]["direction"], "max")
        self.assertEqual(v2["goals"]["total_tokens"]["value"], 4562.0)
        self.assertIn("normalized", v2["goals"]["total_tokens"])
        # Observed (non-goal) metrics present too.
        self.assertIn("cli_calls", v2["observed"])

    def test_initialized_notification_no_response(self):
        self.assertIsNone(self.mcp._handle({"jsonrpc": "2.0", "method": "notifications/initialized"}))

    def test_filters_to_leaf_interface(self):
        # A row for a DIFFERENT interface under the same config must be excluded.
        other = _run_row("v0", "tab", "x", score=0.99, total_tokens=10)
        other["interface"] = "mcp"
        experiments.append_run(self.results_root, self._cfg, other)
        self._env()  # BANTER_INTERFACE=cli
        resp = self.mcp._handle({"jsonrpc": "2.0", "id": 9, "method": "tools/call",
                                 "params": {"name": "normalized_composite", "arguments": {}}})
        import json
        payload = json.loads(resp["result"]["content"][0]["text"])
        # Only the 3 cli versions — the mcp row is filtered out (still best v2).
        self.assertEqual(sorted(payload["versions"]), ["v0", "v1", "v2"])


class SchemaTests(unittest.TestCase):
    def test_exploded_grain_and_values(self):
        for col in ("version", "task", "challenge"):
            self.assertIn(col, experiments.EXPERIMENT_FIELDS)
        for m in experiments.METRICS:
            self.assertIn(m, experiments.EXPERIMENT_FIELDS)   # bare value columns
        self.assertIn("goals", experiments.DESIGN_FIELDS)     # directions encoded here
        self.assertEqual(experiments.EXPERIMENT_FIELDS[-1], "run_dir")


if __name__ == "__main__":
    unittest.main()
