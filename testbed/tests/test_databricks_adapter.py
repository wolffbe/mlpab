"""Unit tests for the Databricks checker's flexible reads + readback robustness.

Two grading bugs motivate these:
  * tasks name a table but never a SCHEMA, so a model may land it anywhere in
    the `workspace` catalog (e.g. `workspace.feature_store.t`); the grader must
    find it by name, preferring the conventional `default` — not hardcode one
    schema (`_resolve_fqn`);
  * a cold SQL warehouse can leave a statement PENDING past the wait cap, which
    must be polled out, not scored as a missing deliverable (`_sql`), and the
    grader should read through a stable warehouse, not race whatever exists
    (`_first_warehouse`).

The checker is built without its lazy SDK import (object.__new__) and driven
through fakes — no databricks-sdk, no live workspace.
"""

import unittest
from types import SimpleNamespace

import pandas as pd

from evals.adapters.databricks import CATALOG, GRADER_WAREHOUSE_NAME, DatabricksChecker


def _wh(id_, name=None, state=None):
    return SimpleNamespace(id=id_, name=name, state=state)


def _resp(state, statement_id="s1", cols=None, rows=None, types=None):
    cols = cols or []
    types = types or [None] * len(cols)
    columns = [
        SimpleNamespace(name=c, type_name=SimpleNamespace(value=t) if t else None, type_text=t)
        for c, t in zip(cols, types)
    ]
    status = SimpleNamespace(state=SimpleNamespace(value=state), error=None)
    manifest = SimpleNamespace(schema=SimpleNamespace(columns=columns))
    result = SimpleNamespace(data_array=rows or [])
    return SimpleNamespace(status=status, manifest=manifest, result=result, statement_id=statement_id)


class _FakeStmtExec:
    """Returns the scripted sequence of responses across execute+get calls."""

    def __init__(self, sequence):
        self._seq = list(sequence)
        self.calls = 0

    def _next(self):
        r = self._seq[min(self.calls, len(self._seq) - 1)]
        self.calls += 1
        return r

    def execute_statement(self, **_):
        return self._next()

    def get_statement(self, _id):
        return self._next()


def _checker(*, warehouses=None, statements=None):
    c = DatabricksChecker.__new__(DatabricksChecker)
    c._fqn_cache = {}
    c._POLL_SECS = 0  # no real sleeps
    c._warehouse = "wh"
    c._w = SimpleNamespace(
        warehouses=SimpleNamespace(list=lambda: list(warehouses or [])),
        statement_execution=_FakeStmtExec(statements or []),
    )
    return c


class WarehouseSelectionTests(unittest.TestCase):
    def test_prefers_named_grader_warehouse(self):
        c = _checker(
            warehouses=[
                _wh("a", name="other", state="RUNNING"),
                _wh("b", name=GRADER_WAREHOUSE_NAME, state="STOPPED"),
            ]
        )
        self.assertEqual(c._first_warehouse(), "b")

    def test_prefers_running_when_no_named(self):
        c = _checker(
            warehouses=[
                _wh("a", name="x", state="STOPPED"),
                _wh("b", name="y", state="RUNNING"),
            ]
        )
        self.assertEqual(c._first_warehouse(), "b")

    def test_falls_back_to_first(self):
        c = _checker(warehouses=[_wh("a", name="x", state="STOPPED")])
        self.assertEqual(c._first_warehouse(), "a")

    def test_empty_raises(self):
        c = _checker(warehouses=[])
        with self.assertRaises(RuntimeError):
            c._first_warehouse()


class SqlPollingTests(unittest.TestCase):
    def test_polls_pending_until_succeeded(self):
        c = _checker(
            statements=[
                _resp("PENDING"),
                _resp("RUNNING"),
                _resp("SUCCEEDED", cols=["n"], rows=[["1"]]),
            ]
        )
        df = c._sql("SELECT 1 AS n")
        self.assertEqual(df["n"].tolist(), ["1"])

    def test_stuck_pending_raises_after_deadline(self):
        # _POLL_SECS=0 → the loop falls back to its fixed iteration cap, runs
        # instantly, then gives up with the terminal (non-success) state.
        c = _checker(statements=[_resp("PENDING")])
        with self.assertRaises(RuntimeError) as cm:
            c._sql("SELECT 1")
        self.assertIn("PENDING", str(cm.exception))


class ResolveFqnTests(unittest.TestCase):
    def _df(self, pairs):
        # pairs: list of (catalog, schema)
        return pd.DataFrame(
            {"table_catalog": [c for c, _ in pairs], "table_schema": [s for _, s in pairs]}
        )

    def test_prefers_workspace_default_when_multiple(self):
        c = _checker()
        c._sql = lambda q: self._df([(CATALOG, "feature_store"), (CATALOG, "default")])
        self.assertEqual(c._resolve_fqn("t"), f"{CATALOG}.default.t")

    def test_finds_non_default_schema(self):
        # the mistral-large case: table only in workspace.feature_store
        c = _checker()
        c._sql = lambda q: self._df([(CATALOG, "feature_store")])
        self.assertEqual(c._resolve_fqn("t"), f"{CATALOG}.feature_store.t")

    def test_finds_table_in_another_catalog(self):
        # model created its own catalog (mistral-large made a `lakebase_fs` one)
        c = _checker()
        c._sql = lambda q: self._df([("proj", "feature_store")])
        self.assertEqual(c._resolve_fqn("t"), "proj.feature_store.t")

    def test_prefers_workspace_over_other_catalog(self):
        c = _checker()
        c._sql = lambda q: self._df([("proj", "default"), (CATALOG, "feature_store")])
        self.assertEqual(c._resolve_fqn("t"), f"{CATALOG}.feature_store.t")

    def test_skips_system_locations(self):
        c = _checker()
        c._sql = lambda q: self._df([("system", "default"), (CATALOG, "feature_store")])
        self.assertEqual(c._resolve_fqn("t"), f"{CATALOG}.feature_store.t")

    def test_missing_returns_none_and_caches(self):
        calls = []

        def fake_sql(q):
            calls.append(q)
            return pd.DataFrame()  # not found in any source

        c = _checker()
        c._sql = fake_sql
        self.assertIsNone(c._resolve_fqn("t"))
        n = len(calls)
        self.assertIsNone(c._resolve_fqn("t"))  # cached — no new _sql calls
        self.assertEqual(len(calls), n)

    def test_read_rows_uses_resolved_fqn(self):
        c = _checker()
        c._resolve_fqn = lambda name: f"proj.feature_store.{name}"
        seen = {}
        c._sql = lambda q: seen.setdefault("q", q) or pd.DataFrame({"x": [1]})
        c.read_rows("t")
        self.assertIn("proj.feature_store.t", seen["q"])


class NumericCastTests(unittest.TestCase):
    def test_numeric_columns_cast_from_strings(self):
        # statement API returns strings; numeric columns must come back numeric
        c = _checker(
            statements=[
                _resp(
                    "SUCCEEDED",
                    cols=["row_id", "event_time", "amount"],
                    rows=[["R0", "1767225600000", "18.88"]],
                    types=["STRING", "LONG", "DOUBLE"],  # Databricks calls BIGINT "LONG"
                )
            ]
        )
        df = c._sql("SELECT *")
        self.assertEqual(df["row_id"].iloc[0], "R0")  # stays string
        self.assertEqual(df["event_time"].iloc[0], 1767225600000)  # int
        self.assertAlmostEqual(df["amount"].iloc[0], 18.88)  # float, not "18.88"
        self.assertTrue(str(df["amount"].dtype).startswith("float"))


class ModelResolveTests(unittest.TestCase):
    def test_candidates_include_default_first_then_other_schemas(self):
        c = _checker()
        c._w.catalogs = SimpleNamespace(list=lambda: [SimpleNamespace(name=CATALOG), SimpleNamespace(name="system")])
        c._w.schemas = SimpleNamespace(
            list=lambda catalog_name=None: [SimpleNamespace(name="default"), SimpleNamespace(name="feature_store")]
        )
        cands = c._model_fqn_candidates("m")
        self.assertEqual(cands[0], f"{CATALOG}.default.m")
        self.assertIn(f"{CATALOG}.feature_store.m", cands)
        self.assertNotIn("system.default.m", cands)  # system catalog skipped


class JobLookupTests(unittest.TestCase):
    def _job(self, jid, name):
        return SimpleNamespace(job_id=jid, settings=SimpleNamespace(name=name))

    def test_falls_back_to_case_insensitive_then_substring(self):
        all_jobs = [self._job(1, "Nightly-FraudJob42"), self._job(2, "other")]

        class Jobs:
            def list(self, name=None):
                if name:  # exact-name server filter misses (case differs)
                    return []
                return all_jobs

            def get(self, jid):
                return next(j for j in all_jobs if j.job_id == jid)

            def list_runs(self, **_):
                return []

        c = _checker()
        c._w.jobs = Jobs()
        out = c.get_job("fraudjob42")  # substring, different case
        self.assertTrue(out["exists"])


class OnlineReadFlagTest(unittest.TestCase):
    def test_databricks_opts_out_of_independent_online_read(self):
        # the online grader skips A3 on databricks because of this flag
        self.assertFalse(DatabricksChecker.supports_online_read)


class ReadRowsVersionTests(unittest.TestCase):
    """A versioned deliverable may keep the bare name or encode the version in
    the name (X_2 / X_vN); read_rows must resolve either."""

    def _read(self, c):
        seen = {}
        c._sql = lambda q: seen.setdefault("q", q) or pd.DataFrame({"x": [1]})
        c.read_rows("t", version=2)
        return seen["q"]

    def test_prefers_version_suffix_over_bare(self):
        # both a bare table and a _v2 exist → the explicit version wins
        c = _checker()
        present = {"t", "t_v2"}
        c._resolve_fqn = lambda n: f"{CATALOG}.default.{n}" if n in present else None
        self.assertIn(f"{CATALOG}.default.t_v2", self._read(c))

    def test_falls_back_to_bare_when_no_suffix(self):
        # only the bare name exists (CREATE OR REPLACE realization) → use it
        c = _checker()
        c._resolve_fqn = lambda n: f"{CATALOG}.default.{n}" if n == "t" else None
        self.assertIn(f"{CATALOG}.default.t", self._read(c))

    def test_accepts_underscore_version_suffix(self):
        c = _checker()
        c._resolve_fqn = lambda n: f"{CATALOG}.default.{n}" if n == "t_2" else None
        self.assertIn(f"{CATALOG}.default.t_2", self._read(c))

    def test_unversioned_reads_bare_only(self):
        c = _checker()
        c._resolve_fqn = lambda n: f"{CATALOG}.default.{n}"
        seen = {}
        c._sql = lambda q: seen.setdefault("q", q) or pd.DataFrame({"x": [1]})
        c.read_rows("t")  # no version
        self.assertIn(f"{CATALOG}.default.t", seen["q"])


class ModelMetricsTests(unittest.TestCase):
    """A5 (capstone) needs the model's training metrics. UC's registry surface
    has none, so get_model reads them from the model version's MLflow run."""

    def _checker_with_uc_model(self, *, run_id, metrics, version=3):
        c = _checker()
        c._model_fqn_candidates = lambda name: [f"{CATALOG}.default.{name}"]
        mv = SimpleNamespace(version=version, run_id=run_id)
        run = {"run": {"data": {"metrics": [{"key": k, "value": v} for k, v in metrics.items()]}}}
        c._w.registered_models = SimpleNamespace(
            get=lambda fqn: SimpleNamespace(full_name=fqn, aliases=[])
        )
        c._w.model_versions = SimpleNamespace(
            get=lambda fn, v: mv,
            list=lambda fn: [SimpleNamespace(version=1, run_id="old"), mv],
        )
        c._w.api_client = SimpleNamespace(do=lambda *a, **k: run)
        return c

    def test_uc_model_metrics_from_linked_run(self):
        c = self._checker_with_uc_model(run_id="r1", metrics={"roc_auc": 0.99, "f1": 0.97})
        out = c.get_model("ccmodel")
        self.assertTrue(out["exists"])
        self.assertEqual(out["metrics"], {"roc_auc": 0.99, "f1": 0.97})

    def test_uc_metrics_empty_when_no_run(self):
        # model exists but has no source run → metrics degrade to {}, no crash
        c = self._checker_with_uc_model(run_id=None, metrics={})
        c._w.model_versions = SimpleNamespace(get=lambda fn, v: SimpleNamespace(version=3, run_id=None), list=lambda fn: [])
        self.assertEqual(c.get_model("m")["metrics"], {})

    def test_run_fetch_failure_degrades_to_empty(self):
        c = self._checker_with_uc_model(run_id="r1", metrics={"roc_auc": 0.9})
        def boom(*a, **k):
            raise RuntimeError("no access")
        c._w.api_client = SimpleNamespace(do=boom)
        out = c.get_model("m")
        self.assertTrue(out["exists"])
        self.assertEqual(out["metrics"], {})


if __name__ == "__main__":
    unittest.main()
