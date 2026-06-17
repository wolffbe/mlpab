"""Unit tests for the two HopsworksChecker grader fixes.

Both were false negatives — the grader marked correct agent work as wrong:
  * get_alert read a wrong REST path; route it through the SDK alerts API.
  * read_training_dataset called get_training_data on a SPLIT dataset, which
    hsfs refuses; fall back to the split getter and reassemble.

The SDK is not installed in the test env, so we build a checker WITHOUT running
__init__ (object.__new__) and attach fakes to `_project` / `_fs`.
"""

import unittest

import pandas as pd

from evals.adapters.hopsworks import HopsworksChecker, _reassemble_split


class FakeAlertsApi:
    def __init__(self, job_alerts=None, alerts=None, job_raises=False):
        self._job_alerts = job_alerts or []
        self._alerts = alerts or []
        self._job_raises = job_raises

    def get_job_alerts(self, name):
        if self._job_raises:
            raise RuntimeError("not supported on this version")
        return [a for a in self._job_alerts if getattr(a, "job_name", None) == name]

    def get_alerts(self):
        return list(self._alerts)


class FakeProject:
    def __init__(self, alerts_api):
        self._alerts_api = alerts_api

    def get_alerts_api(self):
        return self._alerts_api


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _checker_with_project(project):
    c = object.__new__(HopsworksChecker)
    c._project = project
    return c


class GetAlertTests(unittest.TestCase):
    def test_per_job_alert_found(self):
        api = FakeAlertsApi(job_alerts=[_Obj(job_name="flaky36f504")])
        c = _checker_with_project(FakeProject(api))
        out = c.get_alert("flaky36f504")
        self.assertTrue(out["exists"])
        self.assertEqual(out["count"], 1)

    def test_fallback_scans_all_alerts_by_hint(self):
        api = FakeAlertsApi(
            job_alerts=[],
            alerts=[_Obj(name="r", receiver="flaky36f504-failure-receiver", service="jobs")],
        )
        c = _checker_with_project(FakeProject(api))
        out = c.get_alert("flaky36f504")
        self.assertTrue(out["exists"])

    def test_project_scoped_job_alert_counts_without_naming(self):
        # A project-wide job alert covers this job without mentioning it.
        api = FakeAlertsApi(job_alerts=[], alerts=[_Obj(name="r", service="jobs")])
        c = _checker_with_project(FakeProject(api))
        self.assertTrue(c.get_alert("flaky36f504")["exists"])

    def test_no_alert(self):
        api = FakeAlertsApi(job_alerts=[], alerts=[])
        c = _checker_with_project(FakeProject(api))
        self.assertFalse(c.get_alert("flaky36f504")["exists"])

    def test_job_query_unsupported_falls_back(self):
        api = FakeAlertsApi(
            job_raises=True, alerts=[_Obj(name="r", service="jobs", description="flaky36f504")]
        )
        c = _checker_with_project(FakeProject(api))
        self.assertTrue(c.get_alert("flaky36f504")["exists"])


class FakeFeatureView:
    def __init__(self, *, split=None, plain=None):
        self._split = split
        self._plain = plain

    def get_training_data(self, training_dataset_version=1):
        if self._split == 2:
            raise RuntimeError("Incorrect `get` method is used. Use feature_view.get_train_test_split instead.")
        if self._split == 3:
            raise RuntimeError("Use feature_view.get_train_validation_test_split instead.")
        return self._plain

    def get_train_test_split(self, training_dataset_version=1):
        xtr = pd.DataFrame({"f": [1, 2]})
        xte = pd.DataFrame({"f": [3]})
        ytr = pd.DataFrame({"label": [0, 1]})
        yte = pd.DataFrame({"label": [1]})
        return (xtr, xte, ytr, yte)

    def get_train_validation_test_split(self, training_dataset_version=1):
        xs = [pd.DataFrame({"f": [1]}), pd.DataFrame({"f": [2]}), pd.DataFrame({"f": [3]})]
        ys = [pd.DataFrame({"label": [0]}), pd.DataFrame({"label": [1]}), pd.DataFrame({"label": [0]})]
        return (*xs, *ys)


class FakeFS:
    def __init__(self, fv):
        self._fv = fv

    def get_feature_view(self, name):
        return self._fv


def _checker_with_fs(fv):
    c = object.__new__(HopsworksChecker)
    c._fs = FakeFS(fv)
    return c


class ReadTrainingDatasetTests(unittest.TestCase):
    def test_plain_training_data(self):
        plain = (pd.DataFrame({"f": [1, 2]}), pd.DataFrame({"label": [0, 1]}))
        c = _checker_with_fs(FakeFeatureView(plain=plain))
        out = c.read_training_dataset("fv", 1)
        self.assertEqual(len(out), 2)
        self.assertIn("label", out.columns)

    def test_train_test_split_fallback(self):
        c = _checker_with_fs(FakeFeatureView(split=2))
        out = c.read_training_dataset("fv", 1)
        # 2 train + 1 test rows reassembled, features + label joined.
        self.assertEqual(len(out), 3)
        self.assertIn("f", out.columns)
        self.assertIn("label", out.columns)

    def test_train_val_test_split_fallback(self):
        c = _checker_with_fs(FakeFeatureView(split=3))
        out = c.read_training_dataset("fv", 1)
        self.assertEqual(len(out), 3)
        self.assertIn("label", out.columns)

    def test_unrelated_read_error_propagates(self):
        fv = FakeFeatureView(plain=None)

        def boom(training_dataset_version=1):
            raise RuntimeError("real read failure: query service down")

        fv.get_training_data = boom
        c = _checker_with_fs(fv)
        with self.assertRaises(RuntimeError):
            c.read_training_dataset("fv", 1)


class ReassembleSplitTests(unittest.TestCase):
    def test_no_label_split(self):
        parts = (pd.DataFrame({"f": [1, 2]}), pd.DataFrame({"f": [3]}))
        out = _reassemble_split(parts, n_splits=2)
        self.assertEqual(len(out), 3)
        self.assertEqual(list(out.columns), ["f"])


if __name__ == "__main__":
    unittest.main()
