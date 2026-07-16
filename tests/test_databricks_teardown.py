"""Unit tests for the Databricks platform teardown (configs/platforms/databricks/
teardown.py), focused on per-run SCOPING.

With MLPAB_DATABRICKS_PREFIX / MLPAB_DATABRICKS_SCHEMA set the sweep is PER-RUN:
it deletes only this run's resources — the run's UC schema (force-cascade),
resources whose NAME carries the run prefix, and UC-tied resources (serving
endpoints, DLT pipelines) that reference the run schema — so two parallel runs
on one workspace token never tear down each other's work. Unset (or `--all`) the
sweep is FULL: every resource the token user created (the old behaviour, kept as
a janitor). Offline — the module's `_api` is replaced with a fake router; no live
workspace.
"""

import importlib.util
import os
import unittest
from pathlib import Path
from unittest import mock

_TEARDOWN = (
    Path(__file__).resolve().parents[1] / "configs" / "platforms" / "databricks" / "teardown.py"
)

ME = "me@example.com"
OTHER = "someone@else.com"
PREFIX = "mlpabRUN"
SCHEMA = f"workspace.{PREFIX}"


class _FakeApi:
    """Routes the GET reads teardown makes and records every mutating call as
    (method, path, payload)."""

    def __init__(self, **resources):
        self.r = resources
        self.actions: list[tuple] = []

    def __call__(self, method, path, payload=None, query=None):
        if method == "GET":
            return self._get(path, query)
        # POST /experiments/search is a READ despite being POST.
        if method == "POST" and path == "/api/2.0/mlflow/experiments/search":
            return {"experiments": self.r.get("experiments", [])}
        self.actions.append((method, path, payload))
        return {}

    def _get(self, path, query):
        if path == "/api/2.0/preview/scim/v2/Me":
            return {"userName": ME}
        if path == "/api/2.0/serving-endpoints":
            return {"endpoints": self.r.get("serving", [])}
        if path.startswith("/api/2.0/serving-endpoints/"):
            name = path.rsplit("/", 1)[-1]
            return self.r.get("serving_detail", {}).get(name, {})
        if path == "/api/2.0/vector-search/endpoints":
            return {"endpoints": self.r.get("vs", [])}
        if path == "/api/2.1/jobs/runs/list":
            return {"runs": self.r.get("runs", [])}
        if path == "/api/2.0/clusters/list":
            return {"clusters": self.r.get("clusters", [])}
        if path == "/api/2.0/sql/warehouses":
            return {"warehouses": self.r.get("warehouses", [])}
        if path == "/api/2.0/pipelines":
            return {"statuses": self.r.get("pipelines", [])}
        if path.startswith("/api/2.0/pipelines/"):
            pid = path.rsplit("/", 1)[-1]
            return {"spec": self.r.get("pipeline_spec", {}).get(pid, {})}
        if path == "/api/2.1/jobs/list":
            return {"jobs": self.r.get("jobs", [])}
        if path == "/api/2.0/database/instances":
            return {"database_instances": self.r.get("instances", [])}
        if path == "/api/2.1/unity-catalog/tables":
            by = self.r.get("uc_tables_by_schema")
            if by is not None:
                return {"tables": by.get((query or {}).get("schema_name"), [])}
            return {"tables": self.r.get("uc_tables", [])}
        if path == "/api/2.1/unity-catalog/schemas":
            return {"schemas": self.r.get("schemas", [])}
        if path == "/api/2.1/unity-catalog/catalogs":
            return {"catalogs": self.r.get("catalogs", [])}
        if path == "/api/2.0/mlflow/registered-models/search":
            return {"registered_models": self.r.get("models", [])}
        if path == "/api/2.1/unity-catalog/catalogs":
            return {"catalogs": self.r.get("catalogs", [])}
        if path == "/api/2.1/unity-catalog/schemas":
            return {"schemas": self.r.get("schemas", [])}
        if path == "/api/2.0/workspace/list":
            return {"objects": self.r.get("ws_objects", [])}
        if path == "/api/2.0/dbfs/list":
            return {"files": self.r.get("dbfs_files", [])}
        return {}

    # convenience views over recorded mutations
    def deleted_paths(self):
        return [p for m, p, _ in self.actions if m == "DELETE"]

    def posted(self):
        return [(p, pl) for m, p, pl in self.actions if m == "POST"]


def _load(env):
    """Import teardown.py fresh with `env` applied (module reads HOST/TOKEN/
    PREFIX/SCHEMA at import time)."""
    base = {
        "DATABRICKS_HOST": "https://x.cloud.databricks.com",
        "DATABRICKS_TOKEN": "tok",
        "MLPAB_DATABRICKS_PREFIX": "",
        "MLPAB_DATABRICKS_SCHEMA": "",
    }
    base.update(env)
    with mock.patch.dict(os.environ, base, clear=False):
        spec = importlib.util.spec_from_file_location("databricks_teardown", _TEARDOWN)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


class PerRunScopeTests(unittest.TestCase):
    def _resources(self):
        return dict(
            jobs=[
                {"job_id": 1, "settings": {"name": f"{PREFIX}_sync"}},
                {"job_id": 2, "settings": {"name": "other_sync"}, "creator_user_name": ME},
            ],
            serving=[
                {"name": f"{PREFIX}_ep", "creator": ME},
                {"name": "feat_ep", "creator": ME},
                {"name": "foreign_ep", "creator": ME},
            ],
            serving_detail={
                "feat_ep": {
                    "config": {"served_entities": [{"entity_name": f"{SCHEMA}.transactions"}]}
                },
                "foreign_ep": {
                    "config": {"served_entities": [{"entity_name": "workspace.other.x"}]}
                },
            },
            pipelines=[
                {"pipeline_id": "p1", "name": f"{PREFIX}_pipe"},
                {"pipeline_id": "p2", "name": "plain"},
                {"pipeline_id": "p3", "name": "foreign"},
            ],
            pipeline_spec={
                "p2": {"catalog": "workspace", "schema": PREFIX},
                "p3": {"catalog": "workspace", "schema": "other"},
            },
            warehouses=[
                {"id": "w1", "name": f"{PREFIX}_wh"},
                {"id": "w2", "name": "mlpab-grader"},
                {"id": "w3", "name": "someone-wh", "creator_name": ME},
            ],
        )

    def test_deletes_run_schema(self):
        mod = _load({"MLPAB_DATABRICKS_PREFIX": PREFIX, "MLPAB_DATABRICKS_SCHEMA": SCHEMA})
        api = _FakeApi(**self._resources())
        with mock.patch.object(mod, "_api", api):
            mod.main()
        self.assertIn(f"/api/2.1/unity-catalog/schemas/{SCHEMA}", api.deleted_paths())

    def test_jobs_scoped_by_prefix_not_owner(self):
        mod = _load({"MLPAB_DATABRICKS_PREFIX": PREFIX, "MLPAB_DATABRICKS_SCHEMA": SCHEMA})
        api = _FakeApi(**self._resources())
        with mock.patch.object(mod, "_api", api):
            mod.main()
        # job 1 (prefixed) deleted; job 2 (mine but no prefix) kept.
        self.assertIn(("/api/2.1/jobs/delete", {"job_id": 1}), api.posted())
        self.assertNotIn(("/api/2.1/jobs/delete", {"job_id": 2}), api.posted())

    def test_serving_endpoint_matched_by_prefix_or_schema_ref(self):
        mod = _load({"MLPAB_DATABRICKS_PREFIX": PREFIX, "MLPAB_DATABRICKS_SCHEMA": SCHEMA})
        api = _FakeApi(**self._resources())
        with mock.patch.object(mod, "_api", api):
            mod.main()
        deleted = api.deleted_paths()
        self.assertIn(f"/api/2.0/serving-endpoints/{PREFIX}_ep", deleted)  # prefix
        self.assertIn("/api/2.0/serving-endpoints/feat_ep", deleted)  # schema ref
        self.assertNotIn("/api/2.0/serving-endpoints/foreign_ep", deleted)  # neither

    def test_pipeline_matched_by_prefix_or_target_schema(self):
        mod = _load({"MLPAB_DATABRICKS_PREFIX": PREFIX, "MLPAB_DATABRICKS_SCHEMA": SCHEMA})
        api = _FakeApi(**self._resources())
        with mock.patch.object(mod, "_api", api):
            mod.main()
        deleted = api.deleted_paths()
        self.assertIn("/api/2.0/pipelines/p1", deleted)  # prefix
        self.assertIn("/api/2.0/pipelines/p2", deleted)  # target schema ref
        self.assertNotIn("/api/2.0/pipelines/p3", deleted)  # foreign schema

    def test_grader_warehouse_preserved(self):
        mod = _load({"MLPAB_DATABRICKS_PREFIX": PREFIX, "MLPAB_DATABRICKS_SCHEMA": SCHEMA})
        api = _FakeApi(**self._resources())
        with mock.patch.object(mod, "_api", api):
            mod.main()
        deleted = api.deleted_paths()
        self.assertIn("/api/2.0/sql/warehouses/w1", deleted)  # prefixed → swept
        self.assertNotIn("/api/2.0/sql/warehouses/w2", deleted)  # grader → kept
        self.assertNotIn("/api/2.0/sql/warehouses/w3", deleted)  # mine but no prefix → kept

    def test_files_scoped_to_run_subdir(self):
        mod = _load({"MLPAB_DATABRICKS_PREFIX": PREFIX, "MLPAB_DATABRICKS_SCHEMA": SCHEMA})
        api = _FakeApi(**self._resources())
        with mock.patch.object(mod, "_api", api):
            mod.main()
        posted = dict(api.posted())
        self.assertEqual(
            posted.get("/api/2.0/workspace/delete"),
            {"path": f"/Users/{ME}/{PREFIX}", "recursive": True},
        )
        self.assertEqual(
            posted.get("/api/2.0/dbfs/delete"), {"path": f"/FileStore/{PREFIX}", "recursive": True}
        )


class FoundationModelSkipTests(unittest.TestCase):
    """The workspace's hosted LLM/embedding endpoints carry no creator, so the
    full-sweep ownership rule would otherwise delete them. They must be skipped
    in BOTH modes — the agents and grader depend on them."""

    def _resources(self):
        return dict(
            serving=[
                {"name": "databricks-gpt-5", "endpoint_type": "FOUNDATION_MODEL_API"},
                {
                    "name": "embed",
                    "config": {"served_entities": [{"foundation_model": {"name": "x"}}]},
                },
                {"name": f"{PREFIX}_real", "creator": ME},  # a genuine agent endpoint
            ],
        )

    def test_per_run_skips_foundation_models(self):
        mod = _load({"MLPAB_DATABRICKS_PREFIX": PREFIX, "MLPAB_DATABRICKS_SCHEMA": SCHEMA})
        api = _FakeApi(**self._resources())
        with mock.patch.object(mod, "_api", api):
            mod.main()
        deleted = api.deleted_paths()
        self.assertNotIn("/api/2.0/serving-endpoints/databricks-gpt-5", deleted)
        self.assertNotIn("/api/2.0/serving-endpoints/embed", deleted)
        self.assertIn(f"/api/2.0/serving-endpoints/{PREFIX}_real", deleted)

    def test_full_sweep_skips_foundation_models(self):
        # creator unset → ownership rule would mark them mine; the skip protects them.
        mod = _load({})
        api = _FakeApi(**self._resources())
        with mock.patch.object(mod, "_api", api):
            mod.main()
        deleted = api.deleted_paths()
        self.assertNotIn("/api/2.0/serving-endpoints/databricks-gpt-5", deleted)
        self.assertNotIn("/api/2.0/serving-endpoints/embed", deleted)


class LakebaseSweepTests(unittest.TestCase):
    """Lakebase database instances are billed compute the old teardown ignored.
    Per-run deletes by prefix; full-sweep deletes only instances explicitly
    owned by the token user (empty/system-creator instances are preserved)."""

    def _instances(self):
        return dict(
            instances=[
                {"name": f"{PREFIX}-store", "creator": ME},  # this run's
                {"name": "other-store", "creator": ME},  # mine, no prefix
                {"name": "system-default", "creator": None},  # system → never purge
                {"name": "foreign-store", "creator": OTHER},  # someone else's
            ],
        )

    def _deleted_instances(self, api):
        return [p.rsplit("/", 1)[-1] for p in api.deleted_paths() if "/database/instances/" in p]

    def test_per_run_deletes_prefixed_instance_only(self):
        mod = _load({"MLPAB_DATABRICKS_PREFIX": PREFIX, "MLPAB_DATABRICKS_SCHEMA": SCHEMA})
        api = _FakeApi(**self._instances())
        with mock.patch.object(mod, "_api", api):
            mod.main()
        self.assertEqual(self._deleted_instances(api), [f"{PREFIX}-store"])

    def test_full_sweep_deletes_my_instances_preserves_system_and_foreign(self):
        mod = _load({})  # full sweep
        api = _FakeApi(**self._instances())
        with mock.patch.object(mod, "_api", api):
            mod.main()
        deleted = self._deleted_instances(api)
        self.assertIn(f"{PREFIX}-store", deleted)
        self.assertIn("other-store", deleted)
        self.assertNotIn("system-default", deleted)  # empty creator → preserved
        self.assertNotIn("foreign-store", deleted)  # other user → preserved

    def test_instance_delete_omits_force(self):
        # this deployment rejects `force`; purge defaults to hard-delete, so the
        # instance DELETE must carry NO query params.
        mod = _load({"MLPAB_DATABRICKS_PREFIX": PREFIX, "MLPAB_DATABRICKS_SCHEMA": SCHEMA})
        api = _FakeApi(**self._instances())
        seen = {}
        orig = api.__call__

        def spy(method, path, payload=None, query=None):
            if method == "DELETE" and "/database/instances/" in path:
                seen.setdefault("queries", []).append(query)
            return orig(method, path, payload, query)

        with mock.patch.object(mod, "_api", side_effect=spy):
            mod.main()
        self.assertTrue(seen.get("queries"))
        self.assertTrue(all(q is None for q in seen["queries"]))

    def test_deletes_synced_tables_and_database_catalog(self):
        # per-run: FOREIGN tables in the run schema go via the synced-tables API
        # (purge_data), and a prefixed database catalog via the db-catalogs API.
        mod = _load({"MLPAB_DATABRICKS_PREFIX": PREFIX, "MLPAB_DATABRICKS_SCHEMA": SCHEMA})
        api = _FakeApi(
            schemas=[{"name": PREFIX}],  # the run schema (workspace.<PREFIX>)
            uc_tables=[
                {"full_name": f"{SCHEMA}.t_online", "table_type": "FOREIGN"},
                {"full_name": f"{SCHEMA}.plain", "table_type": "MANAGED"},
            ],
            catalogs=[{"name": f"{PREFIX}_fs"}, {"name": "workspace"}, {"name": "fs_other"}],
            instances=[{"name": f"{PREFIX}-store", "creator": ME}],
        )
        with mock.patch.object(mod, "_api", api):
            mod.main()
        deleted = api.deleted_paths()
        self.assertIn(f"/api/2.0/database/synced_tables/{SCHEMA}.t_online", deleted)
        self.assertNotIn(f"/api/2.0/database/synced_tables/{SCHEMA}.plain", deleted)  # not FOREIGN
        self.assertIn(f"/api/2.0/database/catalogs/{PREFIX}_fs", deleted)  # prefixed
        self.assertNotIn("/api/2.0/database/catalogs/fs_other", deleted)  # not this run
        self.assertIn(f"/api/2.0/database/instances/{PREFIX}-store", deleted)

    def test_purges_foreign_table_in_extra_prefixed_schema(self):
        # C1 regression: a FOREIGN table in a PREFIXED schema other than the run
        # schema must still be purged via the synced-tables API — otherwise the
        # schema force-delete fails and orphans the backing instance. The fake
        # returns the table ONLY for the extra schema, so this fails unless the
        # scan covers every prefixed schema (not just the run schema).
        mod = _load({"MLPAB_DATABRICKS_PREFIX": PREFIX, "MLPAB_DATABRICKS_SCHEMA": SCHEMA})
        api = _FakeApi(
            schemas=[{"name": PREFIX}, {"name": f"{PREFIX}_xtra"}, {"name": "default"}],
            uc_tables_by_schema={
                f"{PREFIX}_xtra": [
                    {"full_name": f"workspace.{PREFIX}_xtra.t_online", "table_type": "FOREIGN"}
                ]
            },
        )
        with mock.patch.object(mod, "_api", api):
            mod.main()
        self.assertIn(
            f"/api/2.0/database/synced_tables/workspace.{PREFIX}_xtra.t_online", api.deleted_paths()
        )


class VerifyTests(unittest.TestCase):
    """verify()'s leak predicates must mirror main()'s deletion predicates —
    notably DLT pipelines (billed) must not report clean when leaked."""

    def _verify_out(self, mod):
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = mod.verify()
        return rc, buf.getvalue()

    def test_per_run_flags_leaked_prefixed_pipeline(self):
        mod = _load({"MLPAB_DATABRICKS_PREFIX": PREFIX, "MLPAB_DATABRICKS_SCHEMA": SCHEMA})
        api = _FakeApi(pipelines=[{"pipeline_id": "p1", "name": f"{PREFIX}_pipe"}])
        with mock.patch.object(mod, "_api", api):
            _, out = self._verify_out(mod)
        self.assertIn("dlt-pipeline p1", out)

    def test_unprefixed_pipeline_not_flagged_in_per_run(self):
        mod = _load({"MLPAB_DATABRICKS_PREFIX": PREFIX, "MLPAB_DATABRICKS_SCHEMA": SCHEMA})
        api = _FakeApi(pipelines=[{"pipeline_id": "p9", "name": "someone-else"}])
        with mock.patch.object(mod, "_api", api):
            _, out = self._verify_out(mod)
        self.assertNotIn("dlt-pipeline p9", out)


class FullSweepScopeTests(unittest.TestCase):
    def _resources(self):
        return dict(
            jobs=[
                {"job_id": 1, "settings": {"name": "x"}, "creator_user_name": ME},
                {"job_id": 2, "settings": {"name": "y"}, "creator_user_name": OTHER},
            ],
        )

    def test_unset_prefix_sweeps_by_owner(self):
        mod = _load({})  # no prefix → full sweep
        api = _FakeApi(**self._resources())
        with mock.patch.object(mod, "_api", api):
            mod.main()
        # mine deleted, foreign kept (the original ownership rule).
        self.assertIn(("/api/2.1/jobs/delete", {"job_id": 1}), api.posted())
        self.assertNotIn(("/api/2.1/jobs/delete", {"job_id": 2}), api.posted())

    def test_all_flag_forces_owner_sweep_even_with_prefix(self):
        mod = _load({"MLPAB_DATABRICKS_PREFIX": PREFIX, "MLPAB_DATABRICKS_SCHEMA": SCHEMA})
        api = _FakeApi(**self._resources())
        with mock.patch.object(mod, "_api", api):
            mod.main(all_mode=True)
        # --all ignores the prefix and deletes the owner's job (no schema delete).
        self.assertIn(("/api/2.1/jobs/delete", {"job_id": 1}), api.posted())
        self.assertNotIn(f"/api/2.1/unity-catalog/schemas/{SCHEMA}", api.deleted_paths())


if __name__ == "__main__":
    unittest.main()
