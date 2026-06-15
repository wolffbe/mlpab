"""Unit tests for the three-way assertion tally (pass / fail / skip).

The grading model always enumerates a family's FULL assert suite, so
`total_asserts` is constant per family even on failed / no-deliverable runs.
`skipped` = a check not reached (a prerequisite failed) or not applicable on
this platform; `success` = no failed asserts AND at least one passed.
"""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from evals.common import Suite, grade_table_content, tally


class TallyTests(unittest.TestCase):
    def test_counts_pass_fail_skip(self):
        asserts = [
            {"name": "A1", "status": "pass", "passed": True},
            {"name": "A2", "status": "fail", "passed": False},
            {"name": "A3", "status": "skip", "passed": False},
            {"name": "A4", "status": "skip", "passed": False},
        ]
        t = tally(asserts)
        self.assertEqual(t["asserts_passed"], 1)
        self.assertEqual(t["asserts_failed"], 1)
        self.assertEqual(t["asserts_skipped"], 2)
        self.assertEqual(t["total_asserts"], 4)

    def test_success_requires_no_fail_and_a_pass(self):
        # All green → success.
        self.assertTrue(tally([{"status": "pass"}, {"status": "pass"}])["success"])
        # Any fail → not success, even with passes.
        self.assertFalse(tally([{"status": "pass"}, {"status": "fail"}])["success"])
        # A pass + skips (e.g. platform `none` baseline) → success.
        self.assertTrue(tally([{"status": "pass"}, {"status": "skip"}])["success"])
        # All skipped (nothing verified) → not a success.
        self.assertFalse(tally([{"status": "skip"}, {"status": "skip"}])["success"])

    def test_passed_total_invariant(self):
        t = tally([{"status": "pass"}, {"status": "fail"}, {"status": "skip"}])
        self.assertEqual(
            t["asserts_passed"] + t["asserts_failed"] + t["asserts_skipped"],
            t["total_asserts"],
        )


class SuiteTests(unittest.TestCase):
    def test_skip_is_non_blocking(self):
        s = Suite()
        s.check("A1", True)
        self.assertTrue(s.skip("A2", "not applicable"))  # returns True
        rep = s.report(family="x")
        self.assertTrue(rep["success"])
        self.assertEqual(rep["asserts_skipped"], 1)
        self.assertEqual(rep["total_asserts"], 2)
        self.assertEqual(rep["family"], "x")

    def test_check_records_status_and_legacy_passed(self):
        s = Suite()
        s.check("A1", True, "detail-on-pass-is-dropped")
        s.check("A2", False, "why it failed")
        self.assertEqual(s.asserts[0]["status"], "pass")
        self.assertTrue(s.asserts[0]["passed"])
        self.assertNotIn("detail", s.asserts[0])  # detail only kept on failure
        self.assertEqual(s.asserts[1]["status"], "fail")
        self.assertFalse(s.asserts[1]["passed"])
        self.assertEqual(s.asserts[1]["detail"], "why it failed")

    def test_detail_on_pass_is_kept_when_requested(self):
        # An informative note on a passing check (e.g. "platform metrics differ
        # — not a failure") is retained only with detail_on_pass=True.
        s = Suite()
        s.check("A1", True, "informative note", detail_on_pass=True)
        self.assertEqual(s.asserts[0]["status"], "pass")
        self.assertEqual(s.asserts[0]["detail"], "informative note")

    def test_tally_tolerates_status_less_assert(self):
        # A legacy {name, passed} assert (no status) must not crash tally.
        t = tally([{"name": "A1", "passed": True}, {"name": "A2", "status": "pass"}])
        self.assertEqual(t["total_asserts"], 2)
        self.assertEqual(t["asserts_passed"], 1)  # only the status="pass" one


class ReadCsvOrEmptyTests(unittest.TestCase):
    def test_missing_file_returns_empty(self):
        from evals.common import read_csv_or_empty

        out = read_csv_or_empty(Path(tempfile.mkdtemp()) / "nope.csv")
        self.assertTrue(out.empty)

    def test_zero_byte_file_returns_empty_not_raises(self):
        from evals.common import read_csv_or_empty

        p = Path(tempfile.mkdtemp()) / "empty.csv"
        p.write_text("")  # 0-byte file → pandas EmptyDataError if read directly
        out = read_csv_or_empty(p)  # must NOT raise
        self.assertTrue(out.empty)


class DeliverableExistsProbeTests(unittest.TestCase):
    """runner._deliverable_exists: the first GRADED (non-skip) assert decides
    validity — a leading skip must be stepped over."""

    @staticmethod
    def _valid(asserts):
        from mlpab.runner import _deliverable_exists

        return _deliverable_exists(asserts)

    def test_leading_skip_does_not_force_invalid(self):
        # online/none: A1_table_exists skipped, A2_vectors passes → valid.
        asserts = [
            {"name": "A1_table_exists", "status": "skip", "passed": False},
            {"name": "A2_vectors", "status": "pass", "passed": True},
            {"name": "A3_online_read", "status": "skip", "passed": False},
        ]
        self.assertTrue(self._valid(asserts))

    def test_first_graded_failure_is_invalid(self):
        asserts = [
            {"name": "A1_columns", "status": "fail", "passed": False},
            {"name": "A2_row_count", "status": "skip", "passed": False},
        ]
        self.assertFalse(self._valid(asserts))

    def test_all_skipped_is_invalid(self):
        asserts = [{"name": "A1", "status": "skip", "passed": False}]
        self.assertFalse(self._valid(asserts))


class GradeTableContentTests(unittest.TestCase):
    """grade_table_content always enumerates A1/A2/A3 (size 3)."""

    SPEC = {"columns": ["id", "v"], "sort_cols": ["id"], "int_cols": ["id", "v"]}

    def _instance(self, row_count, digest_val):
        d = Path(tempfile.mkdtemp())
        (d / "solution").mkdir()
        (d / "solution" / "truth.json").write_text(
            json.dumps({"seed": 1, "spec": self.SPEC, "row_count": row_count, "digest": digest_val})
        )
        return d

    def test_missing_deliverable_enumerates_full_suite(self):
        # An empty frame (no deliverable read back) still yields A1/A2/A3.
        inst = self._instance(2, "sha256:whatever")
        rep = grade_table_content("t", inst, pd.DataFrame())
        self.assertEqual(rep["total_asserts"], 3)
        self.assertEqual(rep["asserts_failed"], 1)  # A1 columns
        self.assertEqual(rep["asserts_skipped"], 2)  # A2, A3 not reached
        self.assertFalse(rep["success"])
        self.assertEqual(rep["asserts"][0]["name"], "A1_columns")
        self.assertEqual(rep["asserts"][1]["status"], "skip")
        self.assertEqual(rep["asserts"][2]["status"], "skip")

    def test_wrong_row_count_skips_only_content(self):
        inst = self._instance(99, "sha256:whatever")
        produced = pd.DataFrame({"id": [1, 2], "v": [10, 20]})
        rep = grade_table_content("t", inst, produced)
        self.assertEqual(rep["total_asserts"], 3)
        names = {a["name"]: a["status"] for a in rep["asserts"]}
        self.assertEqual(names["A1_columns"], "pass")
        self.assertEqual(names["A2_row_count"], "fail")
        self.assertEqual(names["A3_content"], "fail")  # content still computed/compared

    def test_correct_deliverable_passes(self):
        from evals.common import canonicalize, digest

        produced = pd.DataFrame({"id": [1, 2], "v": [10, 20]})
        real_digest = digest(canonicalize(produced, self.SPEC))
        inst = self._instance(2, real_digest)
        rep = grade_table_content("t", inst, produced)
        self.assertTrue(rep["success"])
        self.assertEqual(rep["asserts_passed"], 3)
        self.assertEqual(rep["total_asserts"], 3)


if __name__ == "__main__":
    unittest.main()
