"""Unit tests for the capstone FTI evals (ccfraud classification, airquality
regression): metrics, bar calibration, generators (gates + instance shape), and
the hybrid grader's metric/coverage asserts on the `none` platform."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from evals.capstone import common
from evals.capstone.airquality import generate as aq
from evals.capstone.ccfraud import generate as cc


class MetricTests(unittest.TestCase):
    def test_rmse(self):
        self.assertAlmostEqual(common.rmse([1, 2, 3], [1, 2, 3]), 0.0)
        self.assertAlmostEqual(common.rmse([0, 0], [3, 4]), 3.5355339, places=5)

    def test_roc_auc_separation(self):
        y = np.array([0, 0, 1, 1])
        self.assertEqual(common.roc_auc(y, [0.1, 0.2, 0.8, 0.9]), 1.0)  # perfect
        self.assertEqual(common.roc_auc(y, [0.9, 0.8, 0.2, 0.1]), 0.0)  # reversed
        self.assertEqual(common.roc_auc(y, [0.5, 0.5, 0.5, 0.5]), 0.5)  # no signal
        self.assertTrue(np.isnan(common.roc_auc([1, 1], [0.3, 0.7])))  # single class

    def test_calibrate_bar_direction(self):
        y = np.array([0, 0, 1, 1])
        # AUC: higher better -> bar between naive(0.5) and reference(1.0)
        cau = common.calibrate_bar(
            "auc", y, [0.5, 0.5, 0.5, 0.5], [0.1, 0.2, 0.8, 0.9], floor=0.7, margin=0.5
        )
        self.assertTrue(cau["naive"] <= cau["bar"] <= cau["reference"])
        # RMSE: lower better -> bar between reference and naive
        yv = np.array([10.0, 20.0, 30.0, 40.0])
        crm = common.calibrate_bar(
            "rmse",
            yv,
            np.full(4, yv.mean()),
            yv + 1.0,
            floor=common.rmse(yv, np.full(4, yv.mean())),
            margin=0.5,
        )
        self.assertTrue(crm["reference"] <= crm["bar"] <= crm["naive"])


class _GenMixin:
    """Shared structural checks over a generated instance."""

    gen = None  # the generator module
    score_file = None  # the agent-visible scoring CSV
    label_col = None

    def _generate(self, seed):
        out = Path(tempfile.mkdtemp()) / "inst"
        meta = self.gen.generate(seed, out)
        return out, meta

    def test_gates_and_shape(self):
        out, meta = self._generate(7)
        self.addCleanup(shutil.rmtree, out, True)
        truth = json.loads((out / "solution" / "truth.json").read_text())
        # bar sits strictly between the naive baseline and the reference model
        if truth["metric"] == "auc":
            self.assertLess(truth["naive"], truth["bar"])
            self.assertLessEqual(truth["bar"], truth["reference"])
        else:
            self.assertLess(truth["reference"], truth["bar"])
            self.assertLessEqual(truth["bar"], truth["naive"])
        # scoring slice must NOT leak the label to the agent
        score = pd.read_csv(out / "data" / self.score_file)
        self.assertNotIn(self.label_col, score.columns)
        # hidden labels cover exactly the scored rows
        labels = pd.read_csv(out / "solution" / "test_labels.csv")
        self.assertEqual(set(labels[truth["id_col"]]), set(score[truth["id_col"]]))
        self.assertEqual(len(labels), meta["n_test"])

    def test_deterministic(self):
        a, ma = self._generate(3)
        self.addCleanup(shutil.rmtree, a, True)
        b, mb = self._generate(3)
        self.addCleanup(shutil.rmtree, b, True)
        self.assertEqual(ma["bar"], mb["bar"])
        self.assertEqual(ma["record_ids"], mb["record_ids"])


class CcfraudGenTests(_GenMixin, unittest.TestCase):
    gen = cc
    score_file = "score_transactions.csv"
    label_col = "is_fraud"


class AirqualityGenTests(_GenMixin, unittest.TestCase):
    gen = aq
    score_file = "forecast_days.csv"
    label_col = "pm25"


class GradeNoneTests(unittest.TestCase):
    """Hybrid grade on platform=none: only the predictions + metric asserts
    apply (artifact asserts are real-platform only)."""

    def _instance(self, gen):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        attempt = root / "attempt"
        task = attempt / "task"
        (task / "submission").mkdir(parents=True)
        staging = root / "staging"
        gen.generate(7, staging)
        shutil.move(str(staging / "data"), str(task / "data"))
        shutil.move(str(staging / "solution"), str(attempt / "solution"))
        truth = json.loads((attempt / "solution" / "truth.json").read_text())
        labels = pd.read_csv(attempt / "solution" / "test_labels.csv")
        return attempt, task, truth, labels

    def _write(self, task, truth, frame):
        frame.to_csv(task / "submission" / f"{truth['predictions_table']}.csv", index=False)

    def _grade(self, attempt, task):
        return common.grade_capstone(attempt, "none", task)

    def test_perfect_predictions_pass(self):
        attempt, task, truth, labels = self._instance(cc)
        good = labels.rename(columns={"is_fraud": "fraud_probability"})
        good["fraud_probability"] = good["fraud_probability"] * 0.9 + 0.05
        self._write(task, truth, good)
        rep = self._grade(attempt, task)
        self.assertTrue(rep["success"])
        # Full suite always enumerated: A1 table, A1 coverage, A2 metric (graded
        # here) + A3 feature group, A4 training dataset, A5 model (skipped on
        # platform `none`).
        self.assertEqual(rep["total_asserts"], 6)
        self.assertEqual(rep["asserts_passed"], 3)
        self.assertEqual(rep["asserts_skipped"], 3)
        self.assertEqual(rep["asserts_failed"], 0)

    def test_constant_prediction_fails_metric(self):
        attempt, task, truth, labels = self._instance(cc)
        bad = labels[["transaction_id"]].copy()
        bad["fraud_probability"] = 0.5
        self._write(task, truth, bad)
        rep = self._grade(attempt, task)
        self.assertFalse(rep["success"])
        self.assertFalse([a for a in rep["asserts"] if a["name"] == "A2_metric"][0]["passed"])

    def test_incomplete_predictions_fail_coverage(self):
        attempt, task, truth, labels = self._instance(aq)
        partial = labels.iloc[:5].rename(columns={"pm25": "pm25_pred"})
        self._write(task, truth, partial)
        rep = self._grade(attempt, task)
        self.assertFalse(rep["success"])
        self.assertFalse([a for a in rep["asserts"] if a["name"] == "A1_coverage"][0]["passed"])

    def test_airquality_good_regression_passes(self):
        attempt, task, truth, labels = self._instance(aq)
        good = labels.rename(columns={"pm25": "pm25_pred"})
        good["pm25_pred"] = good["pm25_pred"] + 0.5  # tiny error, well under bar
        self._write(task, truth, good)
        rep = self._grade(attempt, task)
        self.assertTrue(rep["success"])


if __name__ == "__main__":
    unittest.main()
