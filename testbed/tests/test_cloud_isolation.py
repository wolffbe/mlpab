"""Unit tests for per-run ISOLATION on aws / gcp / azure (parallel-safe teardown).

The runner mints a per-run id (MLPAB_<P>_PREFIX, and for GCP a per-run
GCP_BQ_DATASET) that the agent stamps on every resource, the grader prepends
when reading back, and teardown scopes its sweep to. These tests cover the three
load-bearing pieces, all OFFLINE (no cloud): the teardown `_scoped`/`_sweep`
selection (per-run deletes only prefixed resources; --all / unset prefix deletes
everything), the adapter `_q` name-prefixing, and the AWS capstone-A5 metrics
read. Mirrors tests/test_databricks_teardown.py / test_databricks_adapter.py.
"""

import importlib.util
import os
import unittest
from pathlib import Path
from unittest import mock

_CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "platforms"
PREFIX = "mlpabRUN"


def _load_teardown(platform, env):
    """Import a platform's teardown.py fresh with `env` applied (the module reads
    PREFIX from os.environ at import time)."""
    path = _CONFIGS / platform / "teardown.py"
    with mock.patch.dict(os.environ, env, clear=False):
        spec = importlib.util.spec_from_file_location(f"{platform}_teardown", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# AWS teardown — botocore-style client; _sweep lists then deletes per name_key,
# scoping by scope_key (default name_key) so ARN-keyed sweeps scope by a name.
# --------------------------------------------------------------------------
class _FakeAwsClient:
    def __init__(self, items):
        self._items = items
        self.deleted = []

    def list_things(self, **kwargs):
        return {"Things": self._items, "NextToken": None}

    def delete_thing(self, **kwargs):
        self.deleted.append(kwargs)


class AwsTeardownScopingTests(unittest.TestCase):
    def _sweep(self, mod, client, **extra):
        mod._sweep(
            client, "thing", "list_things", "Things", "Name",
            "delete_thing", "Name", **extra,
        )

    def test_per_run_deletes_only_prefixed(self):
        mod = _load_teardown("aws", {"MLPAB_AWS_PREFIX": PREFIX})
        c = _FakeAwsClient([{"Name": f"{PREFIX}-mine"}, {"Name": "other-run"}])
        mod._RUN_MODE = True
        self._sweep(mod, c)
        self.assertEqual([d["Name"] for d in c.deleted], [f"{PREFIX}-mine"])

    def test_full_sweep_deletes_all(self):
        mod = _load_teardown("aws", {"MLPAB_AWS_PREFIX": PREFIX})
        c = _FakeAwsClient([{"Name": f"{PREFIX}-mine"}, {"Name": "other-run"}])
        mod._RUN_MODE = False  # --all / no prefix
        self._sweep(mod, c)
        self.assertEqual({d["Name"] for d in c.deleted}, {f"{PREFIX}-mine", "other-run"})

    def test_scope_key_scopes_arn_keyed_sweep_by_name(self):
        # Vector-index sweep deletes by ARN but must scope by the human-readable
        # indexName (which carries the prefix), not the ARN.
        mod = _load_teardown("aws", {"MLPAB_AWS_PREFIX": PREFIX})
        items = [
            {"indexArn": "arn:aws:...:1", "indexName": f"{PREFIX}-idx"},
            {"indexArn": "arn:aws:...:2", "indexName": "foreign-idx"},
        ]
        c = _FakeAwsClient(items)
        c.list_things = lambda **k: {"Things": items, "NextToken": None}
        mod._RUN_MODE = True
        mod._sweep(
            c, "vec", "list_things", "Things", "indexArn",
            "delete_thing", "indexArn", scope_key="indexName",
        )
        self.assertEqual([d["indexArn"] for d in c.deleted], ["arn:aws:...:1"])

    def test_signatures_accept_all_mode(self):
        mod = _load_teardown("aws", {"MLPAB_AWS_PREFIX": PREFIX})
        # main/verify must accept all_mode; we don't run them (no AWS), just check
        # the parameter exists so the __main__ --all wiring stays valid.
        import inspect

        self.assertIn("all_mode", inspect.signature(mod.main).parameters)
        self.assertIn("all_mode", inspect.signature(mod.verify).parameters)


# --------------------------------------------------------------------------
# GCP teardown — aiplatform-style objects with .display_name; _sweep filters by
# display_name prefix.
# --------------------------------------------------------------------------
class _Obj:
    def __init__(self, display_name):
        self.display_name = display_name


class GcpTeardownScopingTests(unittest.TestCase):
    def _run(self, mod, run_mode):
        mod._RUN_MODE = run_mode
        deleted = []
        items = [_Obj(f"{PREFIX}_model"), _Obj("foreign_model")]
        mod._sweep("model", lambda: items, lambda o: deleted.append(o.display_name))
        return deleted

    def test_per_run_deletes_only_prefixed(self):
        mod = _load_teardown("gcp", {"MLPAB_GCP_PREFIX": PREFIX})
        self.assertEqual(self._run(mod, True), [f"{PREFIX}_model"])

    def test_full_sweep_deletes_all(self):
        mod = _load_teardown("gcp", {"MLPAB_GCP_PREFIX": PREFIX})
        self.assertEqual(self._run(mod, False), [f"{PREFIX}_model", "foreign_model"])


# --------------------------------------------------------------------------
# Azure teardown — items with .name; _sweep deletes by name, scoped by prefix.
# --------------------------------------------------------------------------
class AzureTeardownScopingTests(unittest.TestCase):
    def _run(self, mod, run_mode):
        mod._RUN_MODE = run_mode
        deleted = []
        items = [_Obj2(f"{PREFIX}-ep"), _Obj2("foreign-ep")]
        mod._sweep("endpoint", lambda: items, lambda n: deleted.append(n))
        return deleted

    def test_per_run_deletes_only_prefixed(self):
        mod = _load_teardown("azure", {"MLPAB_AZURE_PREFIX": PREFIX})
        self.assertEqual(self._run(mod, True), [f"{PREFIX}-ep"])

    def test_full_sweep_deletes_all(self):
        mod = _load_teardown("azure", {"MLPAB_AZURE_PREFIX": PREFIX})
        self.assertEqual(self._run(mod, False), [f"{PREFIX}-ep", "foreign-ep"])


class _Obj2:
    def __init__(self, name):
        self.name = name


# --------------------------------------------------------------------------
# Adapter name-prefixing (`_q`) — bare when unset, `<prefix><sep><name>` when set.
# --------------------------------------------------------------------------
class AdapterPrefixTests(unittest.TestCase):
    def test_aws_q_hyphen(self):
        from evals.adapters.aws import SageMakerChecker

        c = SageMakerChecker.__new__(SageMakerChecker)
        c._prefix = ""
        self.assertEqual(c._q("transactions"), "transactions")
        c._prefix = PREFIX
        self.assertEqual(c._q("transactions"), f"{PREFIX}-transactions")

    def test_azure_q_hyphen(self):
        from evals.adapters.azure import AzureMLChecker

        c = AzureMLChecker.__new__(AzureMLChecker)
        c._prefix = ""
        self.assertEqual(c._q("transactions"), "transactions")
        c._prefix = PREFIX
        self.assertEqual(c._q("transactions"), f"{PREFIX}-transactions")

    def test_gcp_q_underscore_and_bq_unchanged(self):
        from evals.adapters.gcp import VertexChecker

        c = VertexChecker.__new__(VertexChecker)
        c._prefix = ""
        self.assertEqual(c._q("fraud"), "fraud")
        c._prefix = PREFIX
        self.assertEqual(c._q("fraud"), f"{PREFIX}_fraud")
        # BigQuery feature reads are scoped by the per-run dataset, NOT _q, so
        # the table ref stays bare.
        c.project, c.dataset = "proj", "mlpab_run"
        self.assertEqual(c._table_ref("transactions"), "proj.mlpab_run.transactions")


# --------------------------------------------------------------------------
# AWS capstone-A5 metrics — get_model must surface truthy metrics from the model
# package (CustomerMetadataProperties, else the ModelQuality Statistics URI).
# --------------------------------------------------------------------------
class AwsModelMetricsTests(unittest.TestCase):
    def _checker(self, describe_resp):
        from evals.adapters.aws import SageMakerChecker

        c = SageMakerChecker.__new__(SageMakerChecker)
        c._sm = mock.Mock()
        c._sm.describe_model_package.return_value = describe_resp
        c._s3 = mock.Mock()
        return c

    def test_customer_metadata_preferred(self):
        c = self._checker({"CustomerMetadataProperties": {"auc": "0.91"}})
        self.assertEqual(c._package_metrics([{"ModelPackageArn": "arn:1"}]), {"auc": "0.91"})

    def test_model_quality_uri_marker_when_no_metadata(self):
        c = self._checker(
            {"ModelMetrics": {"ModelQuality": {"Statistics": {"S3Uri": "s3://b/k.json"}}}}
        )
        c._s3.get_object.side_effect = Exception("no access")
        out = c._package_metrics([{"ModelPackageArn": "arn:1"}])
        self.assertEqual(out, {"model_metrics_uri": "s3://b/k.json"})

    def test_empty_when_no_metrics(self):
        c = self._checker({})
        self.assertEqual(c._package_metrics([{"ModelPackageArn": "arn:1"}]), {})

    def test_empty_when_no_packages(self):
        c = self._checker({})
        self.assertEqual(c._package_metrics([]), {})


if __name__ == "__main__":
    unittest.main()
