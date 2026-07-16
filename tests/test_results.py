"""Tests for the results table: session ids, legacy rollups, the row schema."""

import csv
import json
import multiprocessing
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mlpab import results


def _append_bench_row(csv_path: str, i: int) -> None:
    """Module-level worker for the concurrent-append test (spawn-picklable)."""
    row = results.Row(
        started_at=f"2026-06-10T00:00:{i:02d}+00:00",
        run="",
        version="",
        platform="hopsworks",
        interface="cli",
        skills="none",
        category="img",
        task="c1",
        asserts_passed=3,
        total_asserts=4,
        wall_time_s=1.0,
        total_tokens=10,
        cost_usd=0.1,
        run_dir=f"/tmp/run-{i}",
    )
    results.append(Path(csv_path), row, fields=results.RESULTS_FIELDS)


class ConfirmOverwriteTests(unittest.TestCase):
    def test_absent_path_proceeds(self):
        missing = Path(tempfile.mkdtemp()) / "nope"
        self.assertTrue(results.confirm_overwrite(missing, assume_yes=False))

    def test_assume_yes_removes_and_proceeds(self):
        d = Path(tempfile.mkdtemp()) / "run"
        d.mkdir()
        (d / "marker").write_text("x")
        self.assertTrue(results.confirm_overwrite(d, assume_yes=True))
        self.assertFalse(d.exists())  # removed, ready to recreate

    def test_non_tty_without_yes_refuses(self):
        d = Path(tempfile.mkdtemp()) / "run"
        d.mkdir()
        fake_stdin = mock.MagicMock()
        fake_stdin.isatty.return_value = False
        with mock.patch("sys.stdin", fake_stdin):
            self.assertFalse(results.confirm_overwrite(d, assume_yes=False))
        self.assertTrue(d.exists())  # untouched

    def test_tty_decline_keeps_dir(self):
        d = Path(tempfile.mkdtemp()) / "run"
        d.mkdir()
        fake_stdin = mock.MagicMock()
        fake_stdin.isatty.return_value = True
        with mock.patch("sys.stdin", fake_stdin), mock.patch("builtins.input", return_value="n"):
            self.assertFalse(results.confirm_overwrite(d, assume_yes=False))
        self.assertTrue(d.exists())

    def test_tty_accept_removes_dir(self):
        d = Path(tempfile.mkdtemp()) / "run"
        d.mkdir()
        fake_stdin = mock.MagicMock()
        fake_stdin.isatty.return_value = True
        with mock.patch("sys.stdin", fake_stdin), mock.patch("builtins.input", return_value="y"):
            self.assertTrue(results.confirm_overwrite(d, assume_yes=False))
        self.assertFalse(d.exists())


class NextSessionIdTests(unittest.TestCase):
    def test_empty_is_zero(self):
        self.assertEqual(results.next_session_id(Path(tempfile.mkdtemp())), "0")

    def test_counts_leading_int_of_combo_folders(self):
        d = Path(tempfile.mkdtemp())
        (d / "0").mkdir()  # pure-int session dir
        (d / "1_hopsworks_cli_no_skills").mkdir()  # combo folder
        (d / "1_hopsworks_sdk_no_skills").mkdir()  # same session 1
        (d / "notes").mkdir()
        (d / "results.csv").write_text("hdr\n")  # file, ignored
        self.assertEqual(results.next_session_id(d), "2")


class RollUpResultsTests(unittest.TestCase):
    """LEGACY merge into the single global results CSV: per-leaf results.csv
    files are deprecated (the runner appends straight to the global table);
    roll_up folds leftover leaf rows in, deduped by run_dir, one row PER
    EXECUTION (no averaging), flat metric names."""

    def _write_mixed_run(self, run_dir: Path):
        """One config-run CSV holding TWO interface combos (like rq1)."""
        run_dir.mkdir(parents=True)
        fields = results.RESULTS_FIELDS
        with open(run_dir / "results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for iface, passed, calls in (("cli", 3, 4), ("sdk", 4, 9)):
                for task in ("a", "b"):
                    row = {k: "" for k in fields}
                    row.update(
                        task=task,
                        platform="hopsworks",
                        interface=iface,
                        skills="none",
                        n=1,
                        asserts_passed=passed,
                        total_asserts=4,
                        wall_time_s=100,
                        total_tokens=5000,
                        cost_usd=0.5,
                        interface_calls=calls,
                        python_calls=1,
                    )
                    w.writerow(row)

    def test_one_row_per_execution_no_averaging(self):
        root = Path(tempfile.mkdtemp())
        parent = root / "runs"
        self._write_mixed_run(parent / "rq1")
        out = root / "out" / "results.csv"
        rows = results.roll_up_results(parent, out)
        self.assertEqual(len(rows), 4)  # 2 interfaces × 2 tasks — every execution
        with open(out) as f:
            csv_rows = list(csv.DictReader(f))
        self.assertEqual(len(csv_rows), 4)
        # Per-execution identity survives: task + interface per row.
        self.assertEqual({r["task"] for r in csv_rows}, {"a", "b"})
        cli_rows = [r for r in csv_rows if r["interface"] == "cli"]
        self.assertEqual(len(cli_rows), 2)
        self.assertEqual(cli_rows[0]["asserts_passed"], "3")
        self.assertEqual(cli_rows[0]["total_asserts"], "4")
        self.assertEqual(cli_rows[0]["interface_calls"], "4")
        # The whole point: same flat columns as the per-run CSVs — no eng_/res_,
        # no avg_, no run/prev_run/whitelist columns.
        header = list(csv_rows[0].keys())
        self.assertEqual(header, results.RESULTS_FIELDS)
        self.assertFalse(any(c.startswith(("eng_", "res_", "avg_")) for c in header))

    def test_reads_nested_leaf_csvs(self):
        # Leaf CSVs live deep under <config>/<model>/<platform>/<interface>/
        # <version>/<skills>/ — there is no per-config rollup CSV anymore.
        root = Path(tempfile.mkdtemp())
        parent = root / "runs"
        leaf = parent / "rq1" / "opus" / "hopsworks" / "cli" / "v1" / "no-skills"
        self._write_mixed_run(leaf)
        out = root / "out" / "results.csv"
        rows = results.roll_up_results(parent, out)
        self.assertEqual(len(rows), 4)
        self.assertEqual({r["task"] for r in rows}, {"a", "b"})

    def test_repeats_across_runs_numbered(self):
        # The SAME config executed in two run folders → n=1 then n=2.
        root = Path(tempfile.mkdtemp())
        parent = root / "runs"
        self._write_mixed_run(parent / "rq1")
        self._write_mixed_run(parent / "rq1-repeat")
        out = root / "out" / "results.csv"
        rows = results.roll_up_results(parent, out)
        self.assertEqual(len(rows), 8)
        cli_a = [r for r in rows if r["interface"] == "cli" and r["task"] == "a"]
        self.assertEqual([r["n"] for r in cli_a], [1, 2])

    def test_empty_parent_writes_header_only(self):
        root = Path(tempfile.mkdtemp())
        out = root / "out" / "results.csv"
        rows = results.roll_up_results(root / "nope", out)
        self.assertEqual(rows, [])
        self.assertTrue(out.exists())
        with open(out) as f:
            self.assertEqual(f.readline().strip().split(","), results.RESULTS_FIELDS)

    def _write_csv(self, path: Path, rows: list[dict]):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=results.RESULTS_FIELDS)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in results.RESULTS_FIELDS})

    def test_merge_keeps_global_rows_and_dedupes_leaves_by_run_dir(self):
        # The global CSV is maintained per-run by the runner; the merge must
        # KEEP its rows (incl. ones in no leaf) and only add leaf rows whose
        # run_dir is missing from the global table.
        root = Path(tempfile.mkdtemp())
        parent = root / "runs"
        base = dict(
            task="a",
            platform="hopsworks",
            interface="cli",
            skills="none",
            asserts_passed=4,
            total_asserts=4,
        )
        out = parent / "results.csv"
        self._write_csv(
            out,
            [
                {**base, "run_dir": "/runs/a/1", "n": 1},  # also in leaf
                {**base, "run_dir": "/runs/a/2", "n": 2},  # global-only
            ],
        )
        self._write_csv(
            parent / "rq1" / "results.csv",
            [
                {**base, "run_dir": "/runs/a/1", "n": 1},  # dup → skipped
                {**base, "run_dir": "/runs/a/0", "n": 1},  # legacy → added
            ],
        )
        rows = results.roll_up_results(parent, out)
        self.assertEqual([r["run_dir"] for r in rows], ["/runs/a/1", "/runs/a/2", "/runs/a/0"])
        # `n` re-numbered across the merged whole, same identity counts up.
        self.assertEqual([r["n"] for r in rows], [1, 2, 3])
        with open(out) as f:
            self.assertEqual(len(list(csv.DictReader(f))), 3)

    def test_merge_is_idempotent(self):
        # Re-running the merge (every session start) must not duplicate rows.
        root = Path(tempfile.mkdtemp())
        parent = root / "runs"
        base = dict(
            task="a",
            platform="hopsworks",
            interface="cli",
            skills="none",
            asserts_passed=4,
            total_asserts=4,
        )
        self._write_csv(parent / "rq1" / "results.csv", [{**base, "run_dir": "/runs/a/1", "n": 1}])
        out = parent / "results.csv"
        results.roll_up_results(parent, out)
        rows = results.roll_up_results(parent, out)
        self.assertEqual(len(rows), 1)

    def test_same_run_in_leaf_and_config_rollup_merged_once(self):
        # The OLD runner wrote each execution to BOTH a per-leaf CSV and a
        # per-config rollup; rglob sees both copies, so the merge must dedup
        # rows it added THIS pass, not just against the pre-existing global.
        root = Path(tempfile.mkdtemp())
        parent = root / "runs"
        base = dict(
            task="a",
            platform="hopsworks",
            interface="cli",
            skills="none",
            asserts_passed=4,
            total_asserts=4,
        )
        row = {**base, "run_dir": "/runs/a/1", "n": 1}
        self._write_csv(parent / "rq1" / "results.csv", [row])  # old rollup
        self._write_csv(
            parent / "rq1" / "hopsworks" / "cli" / "no-skills" / "results.csv", [row]
        )  # old leaf
        out = parent / "results.csv"
        rows = results.roll_up_results(parent, out)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["n"], 1)

    def test_old_combo_schema_global_rows_dropped(self):
        # A pre-migration global CSV in the old combo-summary schema (no
        # run_dir COLUMN, avg_* data) is not execution rows: drop it instead
        # of renumbering garbage into the table.
        root = Path(tempfile.mkdtemp())
        parent = root / "runs"
        out = parent / "results.csv"
        out.parent.mkdir(parents=True)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["combo", "dir", "avg_score"])
            w.writeheader()
            w.writerow({"combo": "hopsworks_cli", "dir": "rq1", "avg_score": 0.5})
        base = dict(
            task="a",
            platform="hopsworks",
            interface="cli",
            skills="none",
            asserts_passed=4,
            total_asserts=4,
        )
        self._write_csv(parent / "rq1" / "results.csv", [{**base, "run_dir": "/runs/a/1", "n": 1}])
        rows = results.roll_up_results(parent, out)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["run_dir"], "/runs/a/1")

    def test_old_schema_leaf_rows_merge_without_new_columns(self):
        # Legacy leaf CSVs may carry an OLD schema (extra columns, no assert
        # columns). The merge must not KeyError; legacy rows simply leave the
        # new assert columns blank and the unknown columns are not written.
        root = Path(tempfile.mkdtemp())
        parent = root / "runs"
        leaf = parent / "rq1" / "results.csv"
        leaf.parent.mkdir(parents=True)
        old_fields = [
            "config",
            "model",
            "platform",
            "interface",
            "version",
            "skills",
            "task",
            "legacy_sub_task",
            "n",
            "started_at",
            "legacy_valid",
            "legacy_score",
            "wall_time_s",
            "run_dir",
        ]
        with open(leaf, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=old_fields)
            w.writeheader()
            w.writerow(
                {
                    "platform": "hopsworks",
                    "interface": "cli",
                    "skills": "none",
                    "task": "feature",
                    "legacy_sub_task": "drift",
                    "n": 1,
                    "legacy_valid": 1,
                    "legacy_score": 0.5,
                    "wall_time_s": 10,
                    "run_dir": "/runs/legacy/1",
                }
            )
        out = parent / "results.csv"
        rows = results.roll_up_results(parent, out)
        self.assertEqual(len(rows), 1)
        with open(out) as f:
            written = list(csv.DictReader(f))
        self.assertEqual(list(written[0].keys()), results.RESULTS_FIELDS)
        self.assertEqual(written[0]["task"], "feature")  # kept verbatim
        self.assertEqual(written[0]["asserts_passed"], "")  # blank, no migration
        self.assertEqual(written[0]["total_asserts"], "")


class AppendRepeatNumberTests(unittest.TestCase):
    def _row(self, run_dir, asserts_passed=2):
        return results.Row(
            started_at="2026-06-10T00:00:00+00:00",
            run="rq1",
            version="",
            platform="hopsworks",
            interface="cli",
            skills="none",
            category="t",
            task="a",
            asserts_passed=asserts_passed,
            total_asserts=4,
            run_dir=run_dir,
        )

    def test_replacing_non_latest_run_keeps_its_n(self):
        # Re-running attempt 1 while attempts 2 and 3 still hold rows must keep
        # n=1 for the replacement — recounting kept rows would mint a duplicate
        # n=3 and lose n=1.
        out = Path(tempfile.mkdtemp()) / "results.csv"
        for i in (1, 2, 3):
            results.append(out, self._row(f"/runs/a/{i}"), fields=results.RESULTS_FIELDS)
        results.append(out, self._row("/runs/a/1", asserts_passed=4), fields=results.RESULTS_FIELDS)
        with open(out) as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(sorted(int(r["n"]) for r in rows), [1, 2, 3])
        replaced = next(r for r in rows if r["run_dir"] == "/runs/a/1")
        self.assertEqual(replaced["n"], "1")
        self.assertEqual(replaced["asserts_passed"], "4")


class RowSchemaTests(unittest.TestCase):
    """Results CSV schema — single file at `results/results.csv` accumulating
    one row per execution."""

    def test_fields_order_matches_documented_layout(self):
        # Identity first (mirroring the run-folder hierarchy), then work cols,
        # then metrics; `run_dir` last.
        self.assertEqual(
            results.RESULTS_FIELDS[:8],
            ["config", "model", "platform", "interface", "version", "skills", "category", "task"],
        )
        self.assertEqual(results.RESULTS_FIELDS[8:10], ["n", "started_at"])
        # Grading outcome (valid/success) then the assertion-suite tallies.
        self.assertEqual(
            results.RESULTS_FIELDS[10:16],
            [
                "valid",
                "success",
                "asserts_passed",
                "asserts_failed",
                "asserts_skipped",
                "total_asserts",
            ],
        )
        self.assertEqual(results.RESULTS_FIELDS[-1], "run_dir")
        # The legacy schema (and its annotation columns) is gone entirely.
        self.assertFalse(hasattr(results, "FIELDS"))
        for dropped in (
            "hypothesis",
            "change",
            "verdict",
            "verdict_reason",
            "keep",
            "observations",
            "proposed_changes",
        ):
            self.assertNotIn(dropped, results.RESULTS_FIELDS, f"{dropped!r} should be gone")
            self.assertFalse(hasattr(results.Row, dropped))

    def test_append_round_trip(self):
        # Default `append` writes the RESULTS_FIELDS view: `run` surfaces as
        # the `config` column, `interface_ref` as `version`, and the
        # sub-task Row.task as the `task` column.
        out = Path(tempfile.mkdtemp()) / "results.csv"
        row = results.Row(
            started_at="2026-05-28T00:00:00+00:00",
            run="rq1",
            version="",
            platform="hopsworks",
            interface="cli",
            skills="none",
            category="image_classification",
            task="aerial",
            interface_ref="v2",
            model="claude-sonnet-4-6",
            asserts_passed=3,
            total_asserts=4,
            wall_time_s=12.3,
            total_tokens=100,
            cost_usd=0.4,
            run_dir=str(out.parent / "a"),
        )
        results.append(out, row)
        with open(out) as f:
            written = list(csv.DictReader(f))
        self.assertEqual(len(written), 1)
        w = written[0]
        self.assertEqual(list(w.keys()), results.RESULTS_FIELDS)
        self.assertEqual(w["config"], "rq1")
        self.assertEqual(w["version"], "v2")
        self.assertEqual(w["model"], "claude-sonnet-4-6")
        # `task` carries the meaningful task id: the sub-task
        # (Row.task), not the parent FTI category (Row.category).
        self.assertEqual(w["task"], "aerial")
        self.assertEqual(w["interface"], "cli")
        self.assertEqual(w["n"], "1")
        self.assertEqual(w["asserts_passed"], "3")
        self.assertEqual(w["total_asserts"], "4")
        self.assertAlmostEqual(float(w["cost_usd"]), 0.4)
        self.assertAlmostEqual(float(w["wall_time_s"]), 12.3)
        self.assertEqual(w["total_tokens"], "100")

    def test_results_schema_uses_plain_metric_names(self):
        # The results CSV uses plain metric names. The cli/mcp/sdk triple
        # collapses into `interface_calls` (off-interface remainder folds into
        # bash_calls); endpoint columns drop (no endpoint policy configured).
        for dropped in (
            "run",
            "cli_calls",
            "mcp_calls",
            "sdk_calls",
            "whitelist_hits",
            "blacklist_hits",
        ):
            self.assertNotIn(
                dropped, results.RESULTS_FIELDS, f"RESULTS_FIELDS should drop {dropped!r}"
            )
        for kept in (
            "started_at",
            "version",
            "n",
            "task",
            "wall_time_s",
            "rate_limit_wait_s",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cost_usd",
            "asserts_passed",
            "total_asserts",
            "interface_calls",
            "python_calls",
            "bash_calls",
            "run_dir",
        ):
            self.assertIn(kept, results.RESULTS_FIELDS, f"RESULTS_FIELDS should keep {kept!r}")

    def test_results_view_interface_calls_and_bash_fold(self):
        # cli row: interface_calls = its cli_calls; off-interface mcp attempts
        # fold into bash_calls (cli/mcp that is NOT the interface = bash).
        raw = dict(
            interface="cli", cli_calls=7, mcp_calls=2, sdk_calls=0, bash_calls=5, python_calls=3
        )
        view = results._results_view(raw)
        self.assertEqual(view["interface_calls"], "7")
        self.assertEqual(view["bash_calls"], "7")  # 5 bash + 2 off-interface mcp
        self.assertEqual(view["python_calls"], 3)
        # sdk row: interface_calls = sdk_calls.
        view = results._results_view(
            dict(
                interface="sdk", cli_calls=0, mcp_calls=0, sdk_calls=9, bash_calls=1, python_calls=2
            )
        )
        self.assertEqual(view["interface_calls"], "9")
        self.assertEqual(view["bash_calls"], "1")

    def test_results_view_uses_plain_names(self):
        # Row metric fields and the results CSV columns share plain names.
        row = results.Row(
            started_at="2026-05-28T00:00:00+00:00",
            run="",
            version="",
            platform="i",
            interface="t",
            skills="none",
            category="img",
            task="c1",
            asserts_passed=2,
            total_asserts=4,
            wall_time_s=10.0,
            total_tokens=100,
            cost_usd=0.5,
            run_dir="/tmp/x",
        )
        out = Path(tempfile.mkdtemp()) / "results.csv"
        results.append(out, row, fields=results.RESULTS_FIELDS)
        with open(out) as f:
            written = list(csv.DictReader(f))[0]
        self.assertEqual(written["wall_time_s"], "10.0")
        self.assertEqual(written["total_tokens"], "100")
        self.assertAlmostEqual(float(written["cost_usd"]), 0.5)


class ConcurrentAppendTests(unittest.TestCase):
    def test_parallel_processes_lose_no_rows(self):
        # `append` rewrites the whole global CSV (run_dir dedup + n renumber).
        # Parallel treatment sessions (the per-platform rq1 configs) append to
        # the SAME file — without the flock guard, concurrent read-modify-writes
        # drop each other's rows. Every row must survive and the shared-identity
        # repeat counter must come out as a clean 1..N sequence.
        out = Path(tempfile.mkdtemp()) / "results.csv"
        n_procs = 8
        ctx = multiprocessing.get_context("spawn")
        procs = [ctx.Process(target=_append_bench_row, args=(str(out), i)) for i in range(n_procs)]
        for p in procs:
            p.start()
        for p in procs:
            p.join(60)
            self.assertEqual(p.exitcode, 0)
        with open(out) as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), n_procs)
        self.assertEqual({r["run_dir"] for r in rows}, {f"/tmp/run-{i}" for i in range(n_procs)})
        self.assertEqual(sorted(int(r["n"]) for r in rows), list(range(1, n_procs + 1)))


class CommandClassificationTests(unittest.TestCase):
    def _transcript(self, *blocks):
        import json

        p = Path(tempfile.mkdtemp()) / "transcript.jsonl"
        p.write_text(json.dumps({"type": "assistant", "message": {"content": list(blocks)}}) + "\n")
        return p

    def test_skill_tool_counted_as_skill_calls(self):
        tr = self._transcript(
            {"type": "tool_use", "name": "Skill", "input": {"skill": "hops-tip"}},
            {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}},
        )
        counts = results.aggregate_commands(tr)
        self.assertEqual(counts["skill_calls"], 1)
        self.assertEqual(counts["bash_calls"], 1)

    def test_sleep_stats_helper(self):
        self.assertEqual(results._sleep_stats("sleep 60; databricks foo"), (1, 60.0))
        self.assertEqual(results._sleep_stats("echo hi"), (0, 0.0))
        self.assertEqual(results._sleep_stats("sleep 5; x; sleep 10"), (2, 15.0))
        self.assertEqual(results._sleep_stats("sleep 2m"), (1, 120.0))
        self.assertEqual(results._sleep_stats("sleep 0.5"), (1, 0.5))
        # word-boundary guards: not a sleep call
        self.assertEqual(results._sleep_stats("sleepy 5"), (0, 0.0))
        self.assertEqual(results._sleep_stats(""), (0, 0.0))

    def test_sleep_counted_across_buckets(self):
        # A sleep chained onto an interface command classifies as a cli call, yet
        # its sleep must still be counted (scan is independent of the bucket).
        tr = self._transcript(
            {
                "type": "tool_use",
                "name": "Bash",
                "input": {"command": "databricks fg get && sleep 90"},
            },
            {"type": "tool_use", "name": "Bash", "input": {"command": "sleep 10"}},
            {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}},
        )
        counts = results.aggregate_commands(tr, cli_binary="databricks")
        self.assertEqual(counts["sleep_calls"], 2)
        self.assertEqual(counts["sleep_time_s"], 100.0)
        self.assertEqual(counts["cli_calls"], 1)  # the chained databricks command

    def test_sleep_excludes_denied_keeps_failed(self):
        # t1: sleep in a DENIED command (blocked before running → 0s) → excluded.
        # t2: sleep in a command that RAN then failed (bad args) → counted.
        # t3: sleep in a clean command → counted.
        import json

        p = Path(tempfile.mkdtemp()) / "transcript.jsonl"
        events = [
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "Bash",
                            "input": {"command": "sleep 60; aws athena foo"},
                        },
                        {
                            "type": "tool_use",
                            "id": "t2",
                            "name": "Bash",
                            "input": {"command": "sleep 30; aws sagemaker bad-args"},
                        },
                        {
                            "type": "tool_use",
                            "id": "t3",
                            "name": "Bash",
                            "input": {"command": "sleep 5"},
                        },
                    ]
                },
            },
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "is_error": True,
                            "content": "DENIED: aws athena is off-interface",
                        },
                        {
                            "type": "tool_result",
                            "tool_use_id": "t2",
                            "is_error": True,
                            "content": "aws: error: argument operation: Invalid choice",
                        },
                    ]
                },
            },
        ]
        p.write_text("\n".join(json.dumps(e) for e in events) + "\n")
        counts = results.aggregate_commands(p)
        self.assertEqual(counts["sleep_calls"], 2)  # t2 + t3, NOT t1 (denied)
        self.assertEqual(counts["sleep_time_s"], 35.0)
        self.assertEqual(counts["denied_calls"], 1)
        self.assertEqual(counts["failed_commands"], 1)

    def test_workspace_tools_bucketed_explicitly(self):
        # Read/Write/Edit/Glob/Grep/TodoWrite get their own columns; tools
        # outside the map (denied ones like WebFetch) are not bucketed at all.
        tr = self._transcript(
            {"type": "tool_use", "name": "Read", "input": {"file_path": "data/train.csv"}},
            {"type": "tool_use", "name": "Read", "input": {"file_path": "data/test.csv"}},
            {"type": "tool_use", "name": "Write", "input": {"file_path": "train.py"}},
            {"type": "tool_use", "name": "Edit", "input": {"file_path": "train.py"}},
            {"type": "tool_use", "name": "NotebookEdit", "input": {"notebook_path": "x.ipynb"}},
            {"type": "tool_use", "name": "Glob", "input": {"pattern": "*.csv"}},
            {"type": "tool_use", "name": "Grep", "input": {"pattern": "label"}},
            {"type": "tool_use", "name": "TodoWrite", "input": {}},
            {"type": "tool_use", "name": "WebFetch", "input": {"url": "http://x"}},
        )
        counts = results.aggregate_commands(tr)
        self.assertEqual(counts["read_calls"], 2)
        self.assertEqual(counts["write_calls"], 1)
        self.assertEqual(counts["edit_calls"], 2)  # Edit + NotebookEdit
        self.assertEqual(counts["glob_calls"], 1)
        self.assertEqual(counts["grep_calls"], 1)
        self.assertEqual(counts["todo_calls"], 1)
        self.assertNotIn("other_tool_calls", counts)

    def test_failed_and_denied_counted_from_tool_results(self):
        # Errored tool results split into hook rejections (DENIED:) vs real
        # execution failures; successful results count nowhere.
        tr = self._transcript(
            {"type": "tool_use", "name": "Bash", "input": {"command": "hops fg create"}},
        )
        import json

        with open(tr, "a") as f:
            for content, is_err in (
                ("DENIED: local python is off-interface in 'cli' mode.", True),
                ("Traceback (most recent call last): boom", True),
                ("error: unknown flag --primary-key", True),
                ("ok", False),
            ):
                f.write(
                    json.dumps(
                        {
                            "type": "user",
                            "message": {
                                "content": [
                                    {"type": "tool_result", "is_error": is_err, "content": content}
                                ]
                            },
                        }
                    )
                    + "\n"
                )
        counts = results.aggregate_commands(tr)
        self.assertEqual(counts["denied_calls"], 1)
        self.assertEqual(counts["failed_commands"], 2)

    def test_python_buried_in_multi_segment_bash_counts_as_python(self):
        # `BASE=/x; python train.py` — first token is the env assignment,
        # second segment runs python. Should count as python_calls.
        tr = self._transcript(
            {
                "type": "tool_use",
                "name": "Bash",
                "input": {"command": "BASE=/tmp/x\ncd $BASE\npython train.py --epochs 5"},
            },
        )
        counts = results.aggregate_commands(tr)
        self.assertEqual(counts["python_calls"], 1)
        self.assertEqual(counts["bash_calls"], 0)

    def test_python_after_env_var_prefix_counts_as_python(self):
        # `FOO=bar python script.py` — env-var prefix style.
        tr = self._transcript(
            {
                "type": "tool_use",
                "name": "Bash",
                "input": {"command": "PYTHONPATH=/x python -m train"},
            },
        )
        counts = results.aggregate_commands(tr)
        self.assertEqual(counts["python_calls"], 1)

    def test_python_in_bash_dash_c_counts_as_python(self):
        # `bash -c "python train.py"` — recursion into the inner script.
        tr = self._transcript(
            {"type": "tool_use", "name": "Bash", "input": {"command": "bash -c 'python train.py'"}},
        )
        counts = results.aggregate_commands(tr)
        self.assertEqual(counts["python_calls"], 1)

    def test_pure_shell_still_counts_as_bash(self):
        tr = self._transcript(
            {
                "type": "tool_use",
                "name": "Bash",
                "input": {"command": "ls -la /tmp && cat /etc/hostname"},
            },
        )
        counts = results.aggregate_commands(tr)
        self.assertEqual(counts["bash_calls"], 1)
        self.assertEqual(counts["python_calls"], 0)

    def test_find_for_py_files_is_not_python(self):
        # `find . -name '*.py'` shouldn't be counted as a python invocation —
        # *.py here is an argument to find, not an executable.
        tr = self._transcript(
            {"type": "tool_use", "name": "Bash", "input": {"command": "find . -name '*.py'"}},
        )
        counts = results.aggregate_commands(tr)
        self.assertEqual(counts["bash_calls"], 1)
        self.assertEqual(counts["python_calls"], 0)

    def test_cli_still_wins_over_python(self):
        # When BOTH cli and python appear, cli takes priority (interface use
        # is the primary intent we're optimising for).
        tr = self._transcript(
            {
                "type": "tool_use",
                "name": "Bash",
                "input": {"command": "hops fit data && python eval.py"},
            },
        )
        counts = results.aggregate_commands(tr, cli_binary="hops")
        self.assertEqual(counts["cli_calls"], 1)
        self.assertEqual(counts["python_calls"], 0)

    def test_cli_subcommand_allowlist_counts_all_services(self):
        # `cli_subcommand` is a comma-joined allowlist: every allowed service of
        # the binary counts as an interface call; others fall through to bash.
        tr = self._transcript(
            {
                "type": "tool_use",
                "name": "Bash",
                "input": {"command": "aws sagemaker list-training-jobs"},
            },
            {
                "type": "tool_use",
                "name": "Bash",
                "input": {"command": "aws s3 cp data s3://bkt/data --recursive"},
            },
            {
                "type": "tool_use",
                "name": "Bash",
                "input": {
                    "command": "aws sagemaker-runtime invoke-endpoint --endpoint-name e o.json"
                },
            },
            {
                "type": "tool_use",
                "name": "Bash",
                "input": {"command": "aws ec2 describe-instances"},
            },
        )
        counts = results.aggregate_commands(
            tr,
            cli_binary="aws",
            cli_subcommand="sagemaker,sagemaker-runtime,s3",
        )
        self.assertEqual(counts["cli_calls"], 3)
        self.assertEqual(counts["bash_calls"], 1)

    def test_cli_subcommand_requires_exec_position(self):
        # An echoed/printed `aws sagemaker …` is not an interface call — the
        # service match must sit in an EXEC-position segment, mirroring the
        # no-subcommand branch.
        tr = self._transcript(
            {
                "type": "tool_use",
                "name": "Bash",
                "input": {"command": "echo aws sagemaker create-training-job"},
            },
        )
        counts = results.aggregate_commands(tr, cli_binary="aws", cli_subcommand="sagemaker,s3")
        self.assertEqual(counts["cli_calls"], 0)
        self.assertEqual(counts["bash_calls"], 1)

    def test_cli_subcommand_skips_global_options(self):
        # Same rule as the enforcement hook: global options before the service
        # don't hide the entrypoint (`aws --region x sagemaker …` is cli), and
        # an option value equal to an allowed service legitimizes nothing.
        tr = self._transcript(
            {
                "type": "tool_use",
                "name": "Bash",
                "input": {"command": "aws --region us-east-1 sagemaker list-models"},
            },
            {
                "type": "tool_use",
                "name": "Bash",
                "input": {"command": "aws --profile sagemaker ec2 describe-instances"},
            },
        )
        counts = results.aggregate_commands(tr, cli_binary="aws", cli_subcommand="sagemaker,s3")
        self.assertEqual(counts["cli_calls"], 1)
        self.assertEqual(counts["bash_calls"], 1)

    def test_denied_marker_is_line_anchored(self):
        # Echoed remote errors containing `…DENIED:` mid-token (PERMISSION_DENIED:)
        # are FAILURES, not hook denials; the hook marker sits at line start
        # (optionally bulleted by Claude Code's blocked-by-hook formatting).
        import json

        tr = self._transcript(
            {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}},
        )
        with open(tr, "a") as f:
            for content in (
                "grpc error: PERMISSION_DENIED: caller lacks permission",
                "PreToolUse:Bash blocked by hook:\n- DENIED: local python is off-interface",
            ):
                f.write(
                    json.dumps(
                        {
                            "type": "user",
                            "message": {
                                "content": [
                                    {"type": "tool_result", "is_error": True, "content": content}
                                ]
                            },
                        }
                    )
                    + "\n"
                )
        counts = results.aggregate_commands(tr)
        self.assertEqual(counts["denied_calls"], 1)
        self.assertEqual(counts["failed_commands"], 1)

    def test_denied_counted_from_structured_commands_log(self):
        # When the hook's commands.jsonl is available, its `denied: true`
        # records are the source of truth: echoed DENIED text in tool results
        # cannot inflate denied_calls, and the remaining errors are failures.
        import json

        tr = self._transcript(
            {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}},
        )
        with open(tr, "a") as f:
            for content in (
                "DENIED: local python is off-interface in 'cli' mode.",  # real
                "DENIED: looks like a denial but the hook never logged it",  # echo
                "Traceback (most recent call last): boom",
            ):
                f.write(
                    json.dumps(
                        {
                            "type": "user",
                            "message": {
                                "content": [
                                    {"type": "tool_result", "is_error": True, "content": content}
                                ]
                            },
                        }
                    )
                    + "\n"
                )
        log = tr.parent / "commands.jsonl"
        log.write_text(
            json.dumps({"tool_name": "Bash", "category": "bash", "tool_input": {"command": "ls"}})
            + "\n"
            + json.dumps(
                {
                    "tool_name": "Bash",
                    "category": "python",
                    "denied": True,
                    "reason": "DENIED: local python is off-interface",
                    "tool_input": {"command": "python x.py"},
                }
            )
            + "\n"
        )
        counts = results.aggregate_commands(tr, commands_log=log)
        self.assertEqual(counts["denied_calls"], 1)
        self.assertEqual(counts["failed_commands"], 2)

    def test_write_commands_log_preserves_denial_records(self):
        # The rebuild from the transcript must not erase the hook's structured
        # denial records — they are the denied_calls source of truth.
        import json

        tr = self._transcript(
            {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}},
        )
        log = tr.parent / "commands.jsonl"
        log.write_text(
            json.dumps(
                {
                    "tool_name": "Bash",
                    "category": "python",
                    "denied": True,
                    "reason": "DENIED: local python is off-interface",
                    "tool_input": {"command": "python x.py"},
                }
            )
            + "\n"
        )
        results.write_commands_log(tr, log)
        recs = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
        self.assertEqual(len(recs), 2)  # rebuilt `ls` + preserved denial
        self.assertEqual(sum(1 for r in recs if r.get("denied")), 1)


class NoRollingAverageTests(unittest.TestCase):
    def test_no_avg_columns_anywhere(self):
        # Rolling `_avg` columns were removed from the results CSV; the schema
        # doesn't carry them and the recompute machinery is gone.
        self.assertFalse([c for c in results.RESULTS_FIELDS if c.endswith(("_avg", "_avg_s"))])
        self.assertFalse(hasattr(results, "recompute_rolling_averages"))


class CollectClientLogsTests(unittest.TestCase):
    def _run_dir(self):
        return Path(tempfile.mkdtemp())

    def _transcript(self, run_dir, *events):
        p = run_dir / "transcript.jsonl"
        p.write_text("\n".join(json.dumps(e) for e in events) + "\n")
        return p

    def _mcp_log(self, boundary, run_dir, server, *lines):
        slug = results._slug_for(run_dir)
        d = boundary / "Library" / "Caches" / "claude-cli-nodejs" / slug / f"mcp-logs-{server}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "log.jsonl").write_text("\n".join(lines) + "\n")

    def test_mcp_startup_crash_is_surfaced(self):
        import json

        run_dir = self._run_dir()
        boundary = run_dir  # HOME == run dir
        self._mcp_log(
            boundary,
            run_dir,
            "hopsworks",
            json.dumps(
                "Server stderr: Traceback (most recent call last):\n  RuntimeError: Login failed"
            ),
            json.dumps(
                {
                    "timestamp": "T",
                    "debug": "Connection failed after 2749ms: MCP error -32000: Connection closed",
                }
            ),
        )
        res = results.collect_client_logs(
            run_dir=run_dir,
            boundary=boundary,
            interface="mcp",
            platform="hopsworks",
            mcp_servers={"hopsworks": {}},
            transcript_path=run_dir / "missing.jsonl",
        )
        self.assertTrue(res["crashed"])
        self.assertIn("Traceback (most recent call last)", res["markers"])
        text = Path(res["path"]).read_text()
        self.assertTrue(res["path"].endswith("hopsworks_client.logs"))
        self.assertIn("MCP server: hopsworks", text)
        self.assertIn("RuntimeError: Login failed", text)
        self.assertIn("Connection closed", text)

    def test_sdk_traceback_in_transcript_is_captured(self):
        run_dir = self._run_dir()
        tr = self._transcript(
            run_dir,
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "is_error": True,
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Traceback (most recent call last):\n  hsfs.client.exceptions.RestAPIError: boom",
                                }
                            ],
                        },
                    ]
                },
            },
        )
        res = results.collect_client_logs(
            run_dir=run_dir,
            boundary=run_dir,
            interface="sdk",
            platform="hopsworks",
            mcp_servers=None,
            transcript_path=tr,
        )
        self.assertTrue(res["crashed"])
        text = Path(res["path"]).read_text()
        self.assertIn("client errors (agent transcript)", text)
        self.assertIn("RestAPIError: boom", text)

    def test_clean_run_writes_file_without_crash(self):
        run_dir = self._run_dir()
        tr = self._transcript(
            run_dir,
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "content": [{"type": "text", "text": "ok, project created"}],
                        },
                    ]
                },
            },
        )
        res = results.collect_client_logs(
            run_dir=run_dir,
            boundary=run_dir,
            interface="mcp",
            platform="hopsworks",
            mcp_servers={"hopsworks": {}},
            transcript_path=tr,
        )
        self.assertFalse(res["crashed"])
        self.assertTrue(Path(res["path"]).exists())
        self.assertIn("(no server log captured)", Path(res["path"]).read_text())

    def test_none_interface_skips_collection(self):
        run_dir = self._run_dir()
        res = results.collect_client_logs(
            run_dir=run_dir,
            boundary=run_dir,
            interface="none",
            platform="none",
            mcp_servers=None,
            transcript_path=None,
        )
        self.assertFalse(res["crashed"])
        self.assertFalse(Path(res["path"]).exists())


class ToolTimerTests(unittest.TestCase):
    """Arrival-time spans (stream-json events carry no timestamps) and the
    platform/local split of wall time."""

    def _timer(self, ticks):
        from mlpab import claude_runner

        it = iter(ticks)
        return claude_runner.ToolTimer(clock=lambda: next(it))

    @staticmethod
    def _use(tid, name, **inp):
        return json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "tool_use", "id": tid, "name": name, "input": inp}]
                },
            }
        )

    @staticmethod
    def _result(tid):
        return json.dumps(
            {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": tid}]}}
        )

    def test_spans_measure_tool_use_to_tool_result(self):
        t = self._timer([0.0, 5.0, 6.0, 8.5])
        t.observe(self._use("a", "Bash", command="hops fg list"))  # t=0
        t.observe(self._result("a"))  # t=5
        t.observe(self._use("b", "Read", file_path="x.csv"))  # t=6
        t.observe(self._result("b"))  # t=8.5
        spans = t.finalize()
        self.assertEqual([s["seconds"] for s in spans], [5.0, 2.5])
        self.assertEqual(spans[0]["tool_name"], "Bash")

    def test_result_event_closes_open_spans_before_backoff_sleep(self):
        # A `result` event ends one attempt; the rate-limit sleep that follows
        # must not inflate a span left open by a kill/error.
        t = self._timer([0.0, 3.0])
        t.observe(self._use("a", "Bash", command="sleep 999"))  # t=0
        t.observe(json.dumps({"type": "result"}))  # t=3
        spans = t.finalize()
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]["seconds"], 3.0)

    def test_finalize_closes_span_open_at_process_death(self):
        t = self._timer([0.0, 7.0])
        t.observe(self._use("a", "Bash", command="aws sagemaker wait"))
        spans = t.finalize()  # t=7
        self.assertEqual(spans[0]["seconds"], 7.0)

    def test_platform_tool_time_counts_only_the_interface_under_test(self):
        spans = [
            {"tool_name": "Bash", "tool_input": {"command": "hops fg list"}, "seconds": 5.0},
            {"tool_name": "Bash", "tool_input": {"command": "ls data/"}, "seconds": 2.0},
            {"tool_name": "Read", "tool_input": {"file_path": "x.csv"}, "seconds": 1.0},
        ]
        self.assertEqual(results.platform_tool_time(spans, cli_binary="hops"), 5.0)
        # Same spans with NO active cli interface (e.g. the local baseline):
        # nothing is platform time.
        self.assertEqual(results.platform_tool_time(spans), 0.0)

    def test_platform_tool_time_sdk_python(self):
        spans = [
            {
                "tool_name": "Bash",
                "tool_input": {"command": "python -c 'import hopsworks; hopsworks.login()'"},
                "seconds": 4.0,
            },
            {
                "tool_name": "Bash",
                "tool_input": {"command": "python -c 'print(1)'"},
                "seconds": 3.0,
            },
        ]
        self.assertEqual(results.platform_tool_time(spans, sdk_module="hopsworks"), 4.0)


if __name__ == "__main__":
    unittest.main()
