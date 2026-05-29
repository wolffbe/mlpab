"""Tests for the results hierarchy: session ids, combo rollups, autoresearch."""
import csv
import tempfile
import unittest
from pathlib import Path

from banter import results


class NextSessionIdTests(unittest.TestCase):
    def test_empty_is_zero(self):
        self.assertEqual(results.next_session_id(Path(tempfile.mkdtemp())), "0")

    def test_counts_leading_int_of_combo_folders(self):
        d = Path(tempfile.mkdtemp())
        (d / "0").mkdir()                                   # pure-int session dir
        (d / "1_mlkit_cli_no_skills_no_session_v0").mkdir()  # combo folder
        (d / "1_mlkit_sdk_no_skills_no_session_v0").mkdir()  # same session 1
        (d / "notes").mkdir()
        (d / "results.csv").write_text("hdr\n")             # file, ignored
        self.assertEqual(results.next_session_id(d), "2")


def _write_runs(combo_dir: Path, *, task, score, tokens):
    """Drop a minimal benchmark-style CSV (BENCHMARK_FIELDS) for rollup tests."""
    combo_dir.mkdir(parents=True)
    fields = results.BENCHMARK_FIELDS
    with open(combo_dir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for ch in ("a", "b"):
            row = {k: "" for k in fields}
            # Benchmark uses plain (unprefixed) metric names.
            row.update(
                task=task, challenge=ch, interface="hopsworks", type="cli", skills="none",
                score=score, total_tokens=tokens, wall_time_s=100, cost_usd=0.5,
            )
            w.writerow(row)


class RollUpCombosTests(unittest.TestCase):
    def test_one_row_per_combo(self):
        parent = Path(tempfile.mkdtemp()) / "benchmark"
        _write_runs(parent / "0_hopsworks_cli_no_skills_no_session_v0", task="image_classification", score=0.6, tokens=9000)
        _write_runs(parent / "0_hopsworks_sdk_no_skills_no_session_v0", task="image_classification", score=0.7, tokens=7000)
        rows = results.roll_up_combos(parent)
        self.assertEqual(len(rows), 2)
        with open(parent / "results.csv") as f:
            csv_rows = list(csv.DictReader(f))
        self.assertEqual({r["combo"] for r in csv_rows},
                         {"0_hopsworks_cli_no_skills_no_session_v0", "0_hopsworks_sdk_no_skills_no_session_v0"})
        self.assertEqual({r["interface"] for r in csv_rows}, {"hopsworks"})
        self.assertEqual(csv_rows[0]["n_runs"], "2")
        self.assertIn("dir", csv_rows[0])


class RowSchemaTests(unittest.TestCase):
    """Master CSV schema — single file at `results/autoresearch/results.csv`
    accumulating one row per (session, increment, task, challenge)."""

    def test_fields_order_matches_documented_layout(self):
        # `started_at` is first (every row records when the engineer started),
        # then identity, then work cols, then metrics.
        self.assertEqual(results.FIELDS[0], "started_at")
        self.assertEqual(results.FIELDS[1:6], ["run", "version", "interface", "type", "skills"])
        self.assertEqual(results.FIELDS[6:8], ["prev_run", "prev_version"])
        # `incr` is gone — `increment` is now itself a plain integer.
        self.assertNotIn("incr", results.FIELDS)
        self.assertEqual(results.FIELDS[-1], "run_dir")
        for dropped in ("above_median", "any_medal", "gold_medal", "silver_medal",
                        "bronze_medal", "gold_threshold", "silver_threshold",
                        "bronze_threshold", "median_threshold", "is_lower_better",
                        "model", "auth"):
            self.assertNotIn(dropped, results.FIELDS, f"{dropped!r} should be gone")

    def test_append_round_trip(self):
        out = Path(tempfile.mkdtemp()) / "results.csv"
        row = results.Row(
            started_at="2026-05-28T00:00:00+00:00",
            run="7", version="v2",
            interface="mlkit", type="cli", skills="none",
            prev_run="3", prev_version="4",
            task="image_classification", challenge="aerial",
            valid_submission=1, score=0.99, medal="gold",
            eng_wall_time_s=12.3, eng_total_tokens=100, eng_cost_usd=0.4,
            res_wall_time_s=5.0, res_total_tokens=200, res_cost_usd=0.6,
            total_wall_time_s=17.3, total_tokens=300, total_cost=1.0,
            run_dir=str(out.parent / "a"),
        )
        results.append(out, row)
        with open(out) as f:
            written = list(csv.DictReader(f))
        self.assertEqual(len(written), 1)
        w = written[0]
        self.assertEqual(w["run"], "7")
        self.assertEqual(w["version"], "v2")
        self.assertEqual(w["prev_run"], "3")
        self.assertEqual(w["prev_version"], "4")
        self.assertEqual(w["challenge"], "aerial")
        self.assertEqual(w["type"], "cli")
        self.assertAlmostEqual(float(w["score"]), 0.99)
        self.assertEqual(w["medal"], "gold")
        self.assertAlmostEqual(float(w["eng_cost_usd"]), 0.4)
        self.assertAlmostEqual(float(w["res_cost_usd"]), 0.6)
        self.assertAlmostEqual(float(w["total_cost"]), 1.0)

    def test_benchmark_schema_uses_plain_metric_names(self):
        # Benchmark has no researcher, so the eng_*/res_*/total_* prefixes
        # collapse to plain names. Continuation hints survive (a benchmark
        # can be derived from an autoresearch session).
        for dropped in ("run", "version",
                        "eng_wall_time_s", "eng_cost_usd", "eng_total_tokens",
                        "res_wall_time_s", "res_cost_usd", "res_total_tokens",
                        "total_wall_time_s", "total_cost",
                        "hypothesis", "verdict", "keep"):
            self.assertNotIn(dropped, results.BENCHMARK_FIELDS,
                             f"BENCHMARK_FIELDS should drop {dropped!r}")
        for kept in ("started_at", "prev_run", "prev_version", "task", "challenge",
                     "wall_time_s", "input_tokens", "output_tokens", "total_tokens", "cost_usd",
                     "score", "medal", "cli_calls", "run_dir"):
            self.assertIn(kept, results.BENCHMARK_FIELDS,
                          f"BENCHMARK_FIELDS should keep {kept!r}")

    def test_benchmark_view_renames_eng_to_plain(self):
        # The Row dataclass holds `eng_*`; the benchmark CSV view renames
        # those columns to plain names.
        row = results.Row(
            started_at="2026-05-28T00:00:00+00:00",
            run="", version="", interface="i", type="t", skills="none",
            prev_run="", prev_version="",
            task="img", challenge="c1",
            valid_submission=1, score=0.5, medal=None,
            eng_wall_time_s=10.0, eng_total_tokens=100, eng_cost_usd=0.5,
            run_dir="/tmp/x",
        )
        out = Path(tempfile.mkdtemp()) / "results.csv"
        results.append(out, row, fields=results.BENCHMARK_FIELDS)
        with open(out) as f:
            written = list(csv.DictReader(f))[0]
        self.assertEqual(written["wall_time_s"], "10.0")
        self.assertEqual(written["total_tokens"], "100")
        self.assertAlmostEqual(float(written["cost_usd"]), 0.5)
        self.assertNotIn("eng_wall_time_s", written)


class CommandClassificationTests(unittest.TestCase):
    def _transcript(self, *blocks):
        import json
        p = Path(tempfile.mkdtemp()) / "transcript.jsonl"
        p.write_text(json.dumps({"type": "assistant", "message": {"content": list(blocks)}}) + "\n")
        return p

    def test_skill_tool_counted_as_skill_calls(self):
        tr = self._transcript(
            {"type": "tool_use", "name": "Skill", "input": {"skill": "mlkit-tip"}},
            {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}},
        )
        counts = results.aggregate_commands(tr)
        self.assertEqual(counts["skill_calls"], 1)
        self.assertEqual(counts["bash_calls"], 1)
        self.assertEqual(counts["other_tool_calls"], 0)

    def test_python_buried_in_multi_segment_bash_counts_as_python(self):
        # `BASE=/x; python train.py` — first token is the env assignment,
        # second segment runs python. Should count as python_calls.
        tr = self._transcript(
            {"type": "tool_use", "name": "Bash",
             "input": {"command": "BASE=/tmp/x\ncd $BASE\npython train.py --epochs 5"}},
        )
        counts = results.aggregate_commands(tr)
        self.assertEqual(counts["python_calls"], 1)
        self.assertEqual(counts["bash_calls"], 0)

    def test_python_after_env_var_prefix_counts_as_python(self):
        # `FOO=bar python script.py` — env-var prefix style.
        tr = self._transcript(
            {"type": "tool_use", "name": "Bash",
             "input": {"command": "PYTHONPATH=/x python -m train"}},
        )
        counts = results.aggregate_commands(tr)
        self.assertEqual(counts["python_calls"], 1)

    def test_python_in_bash_dash_c_counts_as_python(self):
        # `bash -c "python train.py"` — recursion into the inner script.
        tr = self._transcript(
            {"type": "tool_use", "name": "Bash",
             "input": {"command": "bash -c 'python train.py'"}},
        )
        counts = results.aggregate_commands(tr)
        self.assertEqual(counts["python_calls"], 1)

    def test_pure_shell_still_counts_as_bash(self):
        tr = self._transcript(
            {"type": "tool_use", "name": "Bash",
             "input": {"command": "ls -la /tmp && cat /etc/hostname"}},
        )
        counts = results.aggregate_commands(tr)
        self.assertEqual(counts["bash_calls"], 1)
        self.assertEqual(counts["python_calls"], 0)

    def test_find_for_py_files_is_not_python(self):
        # `find . -name '*.py'` shouldn't be counted as a python invocation —
        # *.py here is an argument to find, not an executable.
        tr = self._transcript(
            {"type": "tool_use", "name": "Bash",
             "input": {"command": "find . -name '*.py'"}},
        )
        counts = results.aggregate_commands(tr)
        self.assertEqual(counts["bash_calls"], 1)
        self.assertEqual(counts["python_calls"], 0)

    def test_cli_still_wins_over_python(self):
        # When BOTH cli and python appear, cli takes priority (interface use
        # is the primary intent we're optimising for).
        tr = self._transcript(
            {"type": "tool_use", "name": "Bash",
             "input": {"command": "mlkit fit data && python eval.py"}},
        )
        counts = results.aggregate_commands(tr, cli_binary="mlkit")
        self.assertEqual(counts["cli_calls"], 1)
        self.assertEqual(counts["python_calls"], 0)


if __name__ == "__main__":
    unittest.main()
