"""Tests for the results hierarchy: session ids + autoresearch rollup."""
import csv
import tempfile
import unittest
from pathlib import Path

from banter import results


class NextSessionIdTests(unittest.TestCase):
    def test_empty_is_zero(self):
        self.assertEqual(results.next_session_id(Path(tempfile.mkdtemp())), "0")

    def test_increments_past_max_ignoring_non_numeric(self):
        d = Path(tempfile.mkdtemp())
        (d / "0").mkdir(); (d / "1").mkdir(); (d / "notes").mkdir()
        self.assertEqual(results.next_session_id(d), "2")


class RollupTests(unittest.TestCase):
    def _write_runs(self, inc_dir: Path, score: float, tokens: int) -> None:
        inc_dir.mkdir(parents=True)
        with open(inc_dir / "results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=results.FIELDS)
            w.writeheader()
            for ch in ("a", "b"):
                row = {k: "" for k in results.FIELDS}
                row.update(
                    challenge_id=ch, interface="hopsworks", mode="cli", skills="none",
                    score=score, total_tokens=tokens, wall_time_s=100, cost_usd=0.5,
                    started_at="2026-05-27T00:00:00",
                )
                w.writerow(row)

    def test_rollup_session_and_top_level(self):
        auto = Path(tempfile.mkdtemp()) / "autoresearch"
        sess = auto / "0"
        self._write_runs(sess / "inc0", 0.60, 9000)
        self._write_runs(sess / "inc1", 0.72, 7000)

        results.rollup_autoresearch(sess)

        # Session: one row per increment.
        with open(sess / "results.csv") as f:
            inc_rows = list(csv.DictReader(f))
        self.assertEqual([r["increment"] for r in inc_rows], ["0", "1"])
        self.assertEqual(inc_rows[1]["avg_score"], "0.7200")

        # Top level: one before/after row for the session.
        with open(auto / "results.csv") as f:
            sess_rows = list(csv.DictReader(f))
        self.assertEqual(len(sess_rows), 1)
        r = sess_rows[0]
        self.assertEqual(r["session"], "0")
        self.assertEqual(r["n_increments"], "2")
        self.assertEqual(r["avg_score_before"], "0.6000")
        self.assertEqual(r["avg_score_after"], "0.7200")
        self.assertEqual(r["interfaces"], "hopsworks/cli")

    def test_rollup_noop_without_increments(self):
        sess = Path(tempfile.mkdtemp()) / "autoresearch" / "0"
        sess.mkdir(parents=True)
        results.rollup_autoresearch(sess)  # must not raise
        self.assertFalse((sess / "results.csv").exists())


if __name__ == "__main__":
    unittest.main()
