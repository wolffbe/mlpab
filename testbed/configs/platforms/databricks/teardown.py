"""Databricks platform teardown — delete the resources a run created.

Run automatically by `mlpab run` (the interface `teardown:` step) at the START
and END of every challenge, so each run — and therefore each autoresearch
version — starts and ends with a clean workspace (same contract as the
hopsworks teardown).

Unlike Hopsworks there is NO project container whose deletion cascades: a
Databricks workspace holds many independent resource types, so cleanup is a
PER-TYPE sweep. There are two modes:

  * PER-RUN (default, when MLPAB_DATABRICKS_PREFIX is set): delete only THIS
    run's resources, so two runs can share one workspace token without their
    start/end teardowns deleting each other's work. The runner mints a per-run
    id (src/mlpab/runner.py) that is both a Unity Catalog SCHEMA name in the
    `workspace` catalog (MLPAB_DATABRICKS_SCHEMA=workspace.mlpab<hex>, this run's
    landing zone for every UC object) and a NAME PREFIX / path segment for the
    resource types that live outside any schema. Selection is two-tier:
      - SCHEMA-derived (robust, no agent cooperation): all UC objects (tables,
        volumes, functions, UC models, online tables, vector indexes — 3-part UC
        names) live in the run schema, so force-deleting that one schema cascades
        them. Feature-serving endpoints and DLT pipelines are matched by their
        served-entity / target reference to the run schema.
      - PREFIX-derived (best-effort, needs the agent to honour the prompt): jobs,
        clusters, vector-search endpoints, legacy MLflow models matched by name
        prefix; experiments and workspace/DBFS files by the `/.../mlpab<hex>/`
        path segment. A billed resource the agent leaves un-prefixed is KEPT —
        run `teardown.py --all` (below) to sweep those.

  * FULL SWEEP (`teardown.py --all`, or when no prefix is set): delete EVERY
    resource the token's own user created — the original user-scoped behaviour,
    kept as a between-batch janitor for resources the per-run sweep leaves behind.
    Where the API reports a creator we only delete resources CREATED BY the
    token's user; resources without creator metadata (workspace files, DBFS,
    MLflow experiments) are scoped to the user's own paths, since the workspace
    behind the .env token is dedicated to the testbed.

Sweeps in COST order — the runner caps each aux step (~60 s), so if the script
is cut short the billable resources are already gone:
  1. model-serving + vector-search endpoints (billed while they exist)
  2. cancel active job runs, permanently delete clusters, SQL warehouses,
     Lakebase database instances (purge drops their synced/foreign tables),
     DLT pipelines (billed while running)
  3. jobs, MLflow experiments + workspace-registry models
  4. Unity Catalog: the run schema (force-cascades) in per-run mode, or own
     catalogs/schemas/objects in full-sweep mode — the `workspace` catalog and
     `default` schema always survive
  5. workspace home contents and DBFS /FileStore contents

Talks to the REST API with the STDLIB ONLY (urllib): the cli run venv carries
just the Go binary — no databricks-sdk — so a pure-python script is the one
shape that works for cli, sdk, and mcp runs alike. Auth is DATABRICKS_HOST +
DATABRICKS_TOKEN from the run env (the same keys every interface uses).

Best-effort by design: invoked via the interface `teardown:` step, whose runner
(`_run_aux`) ignores failures and discards output. Nothing here may raise out of
`main()` — a teardown hiccup must never fail an engineer run.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

HOST = (os.environ.get("DATABRICKS_HOST") or "").rstrip("/")
# DATABRICKS_HOST is commonly set scheme-less (e.g. dbc-xxxx.cloud.databricks.com);
# the CLI/SDK tolerate that but urllib needs an explicit scheme, so add one.
if HOST and "://" not in HOST:
    HOST = "https://" + HOST
TOKEN = os.environ.get("DATABRICKS_TOKEN") or ""

# Per-run identity (set by src/mlpab/runner.py). PREFIX is the run id used as a
# name prefix / path segment; SCHEMA_FQN is `<catalog>.<schema>`, this run's UC
# landing zone. Both empty on a manual invocation → full-sweep mode.
PREFIX = os.environ.get("MLPAB_DATABRICKS_PREFIX") or ""
SCHEMA_FQN = (os.environ.get("MLPAB_DATABRICKS_SCHEMA") or "").strip()
_RUN_CATALOG, _RUN_SCHEMA = ((SCHEMA_FQN.split(".", 1) + [""])[:2]) if SCHEMA_FQN else ("", "")

# Never deleted, even when owned by the token user (default/system plumbing the
# next run relies on — the MCP interface binds the `workspace.default` schema).
KEEP_CATALOGS = {
    "system",
    "samples",
    "main",
    "workspace",
    "hive_metastore",
    "__databricks_internal",
}
# Catalogs not even swept inside (system-managed or not UC-governed).
SKIP_SWEEP_CATALOGS = {"system", "samples", "hive_metastore", "__databricks_internal"}
KEEP_SCHEMAS = {"default", "information_schema"}
# SQL warehouses never swept, even when owned by the token user: the grader
# reads through a stable warehouse setup.py provisions (see
# evals/adapters/databricks.py). Sweeping it would reintroduce the readback
# flake this preservation exists to fix.
KEEP_WAREHOUSES = {"mlpab-grader"}
# Lakebase (managed Postgres) database instances never deleted, even in a full
# sweep. Empty/system-creator instances are already preserved by the
# creator-match guard in _sweep_lakebase; this is for any named exceptions.
KEEP_DB_INSTANCES: set[str] = set()


def _api(method: str, path: str, payload: dict | None = None, query: dict | None = None) -> dict:
    url = HOST + path + (("?" + urllib.parse.urlencode(query)) if query else "")
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        body = resp.read()
    return json.loads(body) if body else {}


def _try(
    method: str, path: str, payload: dict | None = None, query: dict | None = None, label: str = ""
) -> dict | None:
    """Best-effort _api: None on any failure (logged only when labelled)."""
    try:
        return _api(method, path, payload, query)
    except Exception as e:
        if label:
            print(f"[databricks teardown] {label} failed: {e}")
        return None


def _mine(item: dict, me: str, *creator_keys: str) -> bool:
    """True unless the item provably belongs to someone else (hopsworks rule).
    Used in FULL-SWEEP mode (delete everything the token user created)."""
    for key in creator_keys:
        creator = item.get(key)
        if creator and me and creator != me:
            return False
    return True


def _has_prefix(name) -> bool:
    """True if a resource NAME carries this run's prefix (per-run mode)."""
    return bool(name and PREFIX and str(name).startswith(PREFIX))


def _refs_run_schema(text) -> bool:
    """True if `text` references this run's UC schema (e.g. a served-entity name
    `workspace.mlpab<hex>.transactions` or a pipeline target). Catches UC-tied
    resources the agent named without the prefix."""
    return bool(SCHEMA_FQN and text and (SCHEMA_FQN + ".") in str(text))


def _is_model_endpoint(ep: dict) -> bool:
    """A hosted foundation/external MODEL serving endpoint (the workspace's LLM
    and embedding endpoints, e.g. databricks-gpt-5, databricks-gte-large-en) —
    NEVER an agent deliverable, and the agents + grader depend on them. These
    carry no creator, so the full-sweep ownership rule would otherwise try to
    delete them. Mirrors the grader adapter's `is_model` skip
    (evals/adapters/databricks.py:_feature_serving_candidates)."""
    if ep.get("endpoint_type") == "FOUNDATION_MODEL_API":
        return True
    ents = (ep.get("config") or {}).get("served_entities") or []
    return any(e.get("foundation_model") or e.get("external_model") for e in ents)


def _endpoint_refs_run_schema(ep_name: str, ep_item: dict | None = None) -> bool:
    """Does a serving endpoint serve an entity in this run's schema? Feature
    serving serves a FeatureSpec / table whose 3-part name carries the schema.
    Reads the served entities from the LIST item's embedded config when present
    (the common case) and only GETs the endpoint detail as a fallback — a
    per-endpoint GET for every endpoint in the workspace would be slow."""
    cfg = (ep_item or {}).get("config")
    if not cfg:
        cfg = (_try("GET", f"/api/2.0/serving-endpoints/{ep_name}") or {}).get("config")
    ents = (cfg or {}).get("served_entities") or []
    return any(_refs_run_schema(e.get("entity_name")) for e in ents)


def _pipeline_refs_run_schema(pipeline_id: str) -> bool:
    """Does a DLT pipeline target this run's schema/catalog? The pipeline spec
    names a target schema (`schema`/`target`) and optionally a `catalog`."""
    spec = (_try("GET", f"/api/2.0/pipelines/{pipeline_id}") or {}).get("spec") or {}
    cat = spec.get("catalog") or ""
    sch = spec.get("schema") or spec.get("target") or ""
    if cat and sch:
        return f"{cat}.{sch}" == SCHEMA_FQN
    # Single-field target (no catalog) → match the bare schema name.
    return bool(sch) and sch == _RUN_SCHEMA


def _delete_uc_model(full_name: str) -> None:
    quoted = urllib.parse.quote(full_name, safe="")
    versions = _try("GET", f"/api/2.1/unity-catalog/models/{quoted}/versions") or {}
    for v in versions.get("model_versions") or []:
        _try("DELETE", f"/api/2.1/unity-catalog/models/{quoted}/versions/{v.get('version')}")
    _try(
        "DELETE", f"/api/2.1/unity-catalog/models/{quoted}", label=f"delete uc model {full_name!r}"
    )
    print(f"[databricks teardown] deleted uc model {full_name!r}")


def _sweep_schema_objects(catalog: str, schema: str, me: str) -> None:
    """Delete the agent's tables / volumes / functions / models inside a kept
    schema (FULL-SWEEP mode only)."""
    scope = {"catalog_name": catalog, "schema_name": schema}
    for kind, result_key, base, force in (
        ("table", "tables", "/api/2.1/unity-catalog/tables", False),
        ("volume", "volumes", "/api/2.1/unity-catalog/volumes", False),
        ("function", "functions", "/api/2.1/unity-catalog/functions", True),
    ):
        resp = _try("GET", base, query=scope) or {}
        for obj in resp.get(result_key) or []:
            full = obj.get("full_name")
            if not full or not _mine(obj, me, "created_by", "owner"):
                continue
            quoted = urllib.parse.quote(full, safe="")
            query = {"force": "true"} if force else None
            if (
                _try("DELETE", f"{base}/{quoted}", query=query, label=f"delete uc {kind} {full!r}")
                is not None
            ):
                print(f"[databricks teardown] deleted uc {kind} {full!r}")
    resp = _try("GET", "/api/2.1/unity-catalog/models", query=scope) or {}
    for obj in resp.get("registered_models") or []:
        full = obj.get("full_name")
        if full and _mine(obj, me, "created_by", "owner"):
            _delete_uc_model(full)


def _sweep_unity_catalog(me: str) -> None:
    """FULL-SWEEP: own catalogs (force-cascade), own schemas in kept catalogs,
    own objects in kept schemas."""
    catalogs = (_try("GET", "/api/2.1/unity-catalog/catalogs") or {}).get("catalogs") or []
    for cat in catalogs:
        name = cat.get("name")
        if not name or name in SKIP_SWEEP_CATALOGS:
            continue
        # Agent-created catalog → force delete cascades all of its contents.
        if name not in KEEP_CATALOGS and _mine(cat, me, "created_by"):
            if (
                _try(
                    "DELETE",
                    f"/api/2.1/unity-catalog/catalogs/{name}",
                    query={"force": "true"},
                    label=f"delete catalog {name!r}",
                )
                is not None
            ):
                print(f"[databricks teardown] deleted catalog {name!r}")
            continue
        # Kept catalog → sweep agent-created schemas, then objects in kept schemas.
        schemas = (
            _try("GET", "/api/2.1/unity-catalog/schemas", query={"catalog_name": name}) or {}
        ).get("schemas") or []
        for sch in schemas:
            sname = sch.get("name")
            if not sname:
                continue
            if sname not in KEEP_SCHEMAS and _mine(sch, me, "created_by"):
                full = sch.get("full_name") or f"{name}.{sname}"
                if (
                    _try(
                        "DELETE",
                        f"/api/2.1/unity-catalog/schemas/{full}",
                        query={"force": "true"},
                        label=f"delete schema {full!r}",
                    )
                    is not None
                ):
                    print(f"[databricks teardown] deleted schema {full!r}")
            elif sname != "information_schema":
                _sweep_schema_objects(name, sname, me)


def _lakebase_table_scopes(run_mode: bool) -> list[tuple[str, str]]:
    """(catalog, schema) pairs to scan for FOREIGN/synced tables. Per-run: the
    run schema plus any other PREFIXED schema in the run catalog (those are the
    schemas _sweep_unity_catalog_run force-deletes). Full-sweep: every non-system
    schema of every non-system catalog (the janitor catches foreign tables an
    agent left anywhere)."""
    if run_mode:
        cat = _RUN_CATALOG or "workspace"
        schemas = (
            _try("GET", "/api/2.1/unity-catalog/schemas", query={"catalog_name": cat}) or {}
        ).get("schemas") or []
        return [
            (cat, s["name"])
            for s in schemas
            if s.get("name") and (s["name"] == _RUN_SCHEMA or _has_prefix(s["name"]))
        ]
    out: list[tuple[str, str]] = []
    cats = (_try("GET", "/api/2.1/unity-catalog/catalogs") or {}).get("catalogs") or []
    for c in cats:
        cn = c.get("name")
        if not cn or cn in SKIP_SWEEP_CATALOGS:
            continue
        schemas = (
            _try("GET", "/api/2.1/unity-catalog/schemas", query={"catalog_name": cn}) or {}
        ).get("schemas") or []
        out += [(cn, s["name"]) for s in schemas if s.get("name") and s["name"] != "information_schema"]
    return out


def _sweep_lakebase(run_mode: bool, me: str) -> None:
    """Lakebase (managed Postgres online store): delete this run's database
    INSTANCES — always-on billed compute the rest of the sweep can't reach. An
    instance has three layers that must go in order, each with its own API (the
    plain UC table/catalog DELETE rejects them — 400/403):
      1. synced tables   — the FOREIGN/Postgres-backed UC tables (purge_data)
      2. database catalogs — UC catalogs registered to the instance
      3. the instance     — only deletable once 1+2 are gone (no `force`: this
                            deployment rejects it, and purge defaults to hard-delete)
    Runs BEFORE the UC sweep so the run schema's foreign tables are gone before
    the schema force-delete. Absent on non-Lakebase workspaces → best-effort.

    Per-run mode selects by name prefix (instances/catalogs) and by the run's
    prefixed schemas (synced tables); full-sweep requires an explicit creator match for
    instances (NOT the loose _mine rule) so an empty/system-creator instance —
    e.g. a workspace's own `default` — is never purged."""
    # 1) Synced (FOREIGN) tables. Scan EVERY schema this run will try to drop —
    #    not just the run schema: _sweep_unity_catalog_run also force-deletes
    #    extra PREFIXED schemas, and a foreign table left in one would make that
    #    force-delete fail (FOREIGN tables reject the plain UC DELETE) and orphan
    #    the backing instance. Full-sweep scans all non-system schemas so the
    #    janitor catches foreign tables wherever an agent placed them.
    for cat, sch in _lakebase_table_scopes(run_mode):
        resp = _try(
            "GET",
            "/api/2.1/unity-catalog/tables",
            query={"catalog_name": cat, "schema_name": sch},
        ) or {}
        for t in resp.get("tables") or []:
            full = t.get("full_name")
            if full and t.get("table_type") == "FOREIGN":
                if (
                    _try("DELETE", f"/api/2.0/database/synced_tables/{full}", query={"purge_data": "true"})
                    is not None
                ):
                    print(f"[databricks teardown] deleted lakebase synced table {full!r}")
    # 2) Database catalogs registered to instances (reject plain UC delete).
    cats = (_try("GET", "/api/2.1/unity-catalog/catalogs") or {}).get("catalogs") or []
    for c in cats:
        cn = c.get("name")
        if not cn or cn in KEEP_CATALOGS:
            continue
        keep = _has_prefix(cn) if run_mode else _mine(c, me, "created_by")
        if keep and _try("DELETE", f"/api/2.0/database/catalogs/{cn}") is not None:
            print(f"[databricks teardown] deleted lakebase catalog {cn!r}")
    # 3) The instances themselves.
    insts = (_try("GET", "/api/2.0/database/instances") or {}).get("database_instances") or []
    for inst in insts:
        name = inst.get("name")
        if not name or name in KEEP_DB_INSTANCES:
            continue
        keep = _has_prefix(name) if run_mode else (inst.get("creator") == me and bool(me))
        if not keep:
            continue
        if (
            _try(
                "DELETE",
                f"/api/2.0/database/instances/{name}",
                label=f"delete lakebase instance {name!r}",
            )
            is not None
        ):
            print(f"[databricks teardown] deleted lakebase instance {name!r}")


def _sweep_unity_catalog_run() -> None:
    """PER-RUN: delete this run's schema (force-cascade), plus any catalog or
    extra schema whose name carries the run prefix. Everything else — including
    `workspace.default` — is left for other runs / the full sweep."""
    if SCHEMA_FQN:
        if (
            _try(
                "DELETE",
                f"/api/2.1/unity-catalog/schemas/{SCHEMA_FQN}",
                query={"force": "true"},
                label=f"delete run schema {SCHEMA_FQN!r}",
            )
            is not None
        ):
            print(f"[databricks teardown] deleted run schema {SCHEMA_FQN!r}")
    # Prefixed catalogs the agent may have created itself (cascades their contents).
    catalogs = (_try("GET", "/api/2.1/unity-catalog/catalogs") or {}).get("catalogs") or []
    for cat in catalogs:
        name = cat.get("name")
        if name and name not in KEEP_CATALOGS and _has_prefix(name):
            if (
                _try(
                    "DELETE",
                    f"/api/2.1/unity-catalog/catalogs/{name}",
                    query={"force": "true"},
                    label=f"delete catalog {name!r}",
                )
                is not None
            ):
                print(f"[databricks teardown] deleted catalog {name!r}")
    # Extra prefixed schemas in the workspace catalog (besides the run schema).
    schemas = (
        _try("GET", "/api/2.1/unity-catalog/schemas", query={"catalog_name": _RUN_CATALOG or "workspace"})
        or {}
    ).get("schemas") or []
    for sch in schemas:
        sname = sch.get("name")
        full = sch.get("full_name") or f"{_RUN_CATALOG or 'workspace'}.{sname}"
        if sname and sname not in KEEP_SCHEMAS and full != SCHEMA_FQN and _has_prefix(sname):
            if (
                _try(
                    "DELETE",
                    f"/api/2.1/unity-catalog/schemas/{full}",
                    query={"force": "true"},
                    label=f"delete schema {full!r}",
                )
                is not None
            ):
                print(f"[databricks teardown] deleted schema {full!r}")


def main(all_mode: bool = False) -> None:
    if not HOST or not TOKEN:
        print("[databricks teardown] DATABRICKS_HOST/DATABRICKS_TOKEN unset — nothing to do")
        return

    who = _try("GET", "/api/2.0/preview/scim/v2/Me", label="identify user")
    if not who:  # workspace unreachable / bad token → nothing we can clean
        return
    me = who.get("userName") or ""

    run_mode = bool(PREFIX) and not all_mode
    if run_mode:
        print(
            f"[databricks teardown] per-run mode: prefix {PREFIX!r}, schema {SCHEMA_FQN!r}"
        )
    else:
        why = "forced (--all)" if all_mode else "no run prefix set"
        print(f"[databricks teardown] full-sweep mode ({why}): user {me!r}")

    # 1) Billed-while-they-exist endpoints.
    resp = _try("GET", "/api/2.0/serving-endpoints", label="list serving endpoints") or {}
    for ep in resp.get("endpoints") or []:
        name = ep.get("name")
        if not name or _is_model_endpoint(ep):
            continue  # never delete the workspace's hosted LLM/embedding models
        delete = (
            (_has_prefix(name) or _endpoint_refs_run_schema(name, ep))
            if run_mode
            else _mine(ep, me, "creator")
        )
        if delete:
            if (
                _try(
                    "DELETE",
                    f"/api/2.0/serving-endpoints/{name}",
                    label=f"delete serving endpoint {name!r}",
                )
                is not None
            ):
                print(f"[databricks teardown] deleted serving endpoint {name!r}")
    resp = _try("GET", "/api/2.0/vector-search/endpoints") or {}
    for ep in resp.get("endpoints") or []:
        name = ep.get("name")
        if not name:
            continue
        delete = _has_prefix(name) if run_mode else _mine(ep, me, "creator")
        if delete:
            if _try("DELETE", f"/api/2.0/vector-search/endpoints/{name}") is not None:
                print(f"[databricks teardown] deleted vector-search endpoint {name!r}")

    # 2) Billed-while-running compute.
    token = None
    while True:
        query = {"active_only": "true"}
        if token:
            query["page_token"] = token
        resp = _try("GET", "/api/2.1/jobs/runs/list", query=query) or {}
        for run in resp.get("runs") or []:
            delete = (
                _has_prefix(run.get("run_name")) if run_mode else _mine(run, me, "creator_user_name")
            )
            if run.get("run_id") and delete:
                _try("POST", "/api/2.1/jobs/runs/cancel", {"run_id": run["run_id"]})
                print(f"[databricks teardown] cancelled run {run['run_id']}")
        token = resp.get("next_page_token")
        if not token:
            break

    resp = _try("GET", "/api/2.0/clusters/list", label="list clusters") or {}
    for cl in resp.get("clusters") or []:
        if cl.get("cluster_source") == "JOB":  # ephemeral; dies with its run
            continue
        delete = (
            _has_prefix(cl.get("cluster_name")) if run_mode else _mine(cl, me, "creator_user_name")
        )
        if cl.get("cluster_id") and delete:
            if (
                _try(
                    "POST",
                    "/api/2.0/clusters/permanent-delete",
                    {"cluster_id": cl["cluster_id"]},
                    label=f"delete cluster {cl['cluster_id']}",
                )
                is not None
            ):
                print(
                    f"[databricks teardown] deleted cluster {cl['cluster_id']} "
                    f"({cl.get('cluster_name')!r})"
                )

    resp = _try("GET", "/api/2.0/sql/warehouses") or {}
    for wh in resp.get("warehouses") or []:
        if wh.get("name") in KEEP_WAREHOUSES:
            continue
        delete = _has_prefix(wh.get("name")) if run_mode else _mine(wh, me, "creator_name")
        if wh.get("id") and delete:
            if _try("DELETE", f"/api/2.0/sql/warehouses/{wh['id']}") is not None:
                print(
                    f"[databricks teardown] deleted sql warehouse {wh['id']} ({wh.get('name')!r})"
                )

    # Lakebase database instances (billed compute; purge drops their synced /
    # foreign tables) — before the UC sweep so the foreign catalogs can clear.
    _sweep_lakebase(run_mode, me)

    token = None
    while True:
        query = {"max_results": "100"}
        if token:
            query["page_token"] = token
        resp = _try("GET", "/api/2.0/pipelines", query=query) or {}
        for pl in resp.get("statuses") or []:
            pid = pl.get("pipeline_id")
            if not pid:
                continue
            delete = (
                (_has_prefix(pl.get("name")) or _pipeline_refs_run_schema(pid))
                if run_mode
                else _mine(pl, me, "creator_user_name")
            )
            if delete:
                if _try("DELETE", f"/api/2.0/pipelines/{pid}") is not None:
                    print(f"[databricks teardown] deleted dlt pipeline {pid}")
        token = resp.get("next_page_token")
        if not token:
            break

    # 3) Jobs + MLflow metadata.
    token = None
    while True:
        query = {"limit": "100"}
        if token:
            query["page_token"] = token
        resp = _try("GET", "/api/2.1/jobs/list", query=query, label="list jobs") or {}
        for job in resp.get("jobs") or []:
            jname = (job.get("settings") or {}).get("name")
            delete = _has_prefix(jname) if run_mode else _mine(job, me, "creator_user_name")
            if job.get("job_id") and delete:
                if _try("POST", "/api/2.1/jobs/delete", {"job_id": job["job_id"]}) is not None:
                    print(f"[databricks teardown] deleted job {job['job_id']}")
        token = resp.get("next_page_token")
        if not token:
            break

    home = f"/Users/{me}"
    run_home = f"{home}/{PREFIX}"  # per-run subdir for files / experiments
    resp = _try("POST", "/api/2.0/mlflow/experiments/search", {"max_results": 1000}) or {}
    for exp in resp.get("experiments") or []:
        # Experiments carry no creator field; a path scope stands in for it —
        # the run subdir in per-run mode, the whole home in full-sweep mode.
        ename = exp.get("name") or ""
        scope = run_home + "/" if run_mode else home + "/"
        if ename.startswith(scope) and exp.get("experiment_id"):
            if (
                _try(
                    "POST",
                    "/api/2.0/mlflow/experiments/delete",
                    {"experiment_id": exp["experiment_id"]},
                )
                is not None
            ):
                print(f"[databricks teardown] deleted experiment {ename!r}")

    resp = (
        _try("GET", "/api/2.0/mlflow/registered-models/search", query={"max_results": "1000"}) or {}
    )
    for rm in resp.get("registered_models") or []:
        delete = _has_prefix(rm.get("name")) if run_mode else _mine(rm, me, "user_id")
        if rm.get("name") and delete:
            if (
                _try("DELETE", "/api/2.0/mlflow/registered-models/delete", {"name": rm["name"]})
                is not None
            ):
                print(f"[databricks teardown] deleted registered model {rm['name']!r}")

    # 4) Unity Catalog (absent on non-UC workspaces — every call is best-effort).
    if run_mode:
        _sweep_unity_catalog_run()
    else:
        _sweep_unity_catalog(me)

    # 5) Files: workspace home + DBFS /FileStore. Per-run mode deletes only this
    #    run's subdir; full-sweep deletes every child (the dirs themselves stay —
    #    they are default plumbing).
    if run_mode:
        # No label: the run subdir often does not exist (the agent created no
        # workspace files), so a 404 here is the normal case, not a failure.
        if _try("POST", "/api/2.0/workspace/delete", {"path": run_home, "recursive": True}) is not None:
            print(f"[databricks teardown] deleted workspace path {run_home!r}")
        store = f"/FileStore/{PREFIX}"
        if _try("POST", "/api/2.0/dbfs/delete", {"path": store, "recursive": True}) is not None:
            print(f"[databricks teardown] deleted dbfs path {store!r}")
    else:
        resp = _try("GET", "/api/2.0/workspace/list", query={"path": home}) or {}
        for obj in resp.get("objects") or []:
            path = obj.get("path")
            if (
                path
                and _try(
                    "POST",
                    "/api/2.0/workspace/delete",
                    {"path": path, "recursive": True},
                    label=f"delete workspace path {path!r}",
                )
                is not None
            ):
                print(f"[databricks teardown] deleted workspace path {path!r}")

        resp = _try("GET", "/api/2.0/dbfs/list", query={"path": "/FileStore"}) or {}
        for f in resp.get("files") or []:
            path = f.get("path")
            if (
                path
                and _try("POST", "/api/2.0/dbfs/delete", {"path": path, "recursive": True})
                is not None
            ):
                print(f"[databricks teardown] deleted dbfs path {path!r}")

    print("[databricks teardown] done")


def verify(all_mode: bool = False) -> int:
    """Twofold teardown check (`teardown.py verify`): (1) the workspace CONNECTS,
    then (2) NONE of the agent's billed resources survive the sweep. Read-only.

    In per-run mode it re-scans only THIS run's cost-bearing resources (the same
    selection main() deletes); with --all (or no prefix) it re-scans every
    user-owned resource. Exit 0 iff clean; non-zero listing the leaks otherwise
    (the runner WARNS rather than failing — the run already happened, and the
    next run's start-teardown sweeps again). UC absence or an unreachable type is
    ignored (best-effort, like main())."""
    if not HOST or not TOKEN:
        print("[databricks verify-teardown] DATABRICKS_HOST/DATABRICKS_TOKEN unset")
        return 1
    who = _try("GET", "/api/2.0/preview/scim/v2/Me")
    if not who:
        print(f"[databricks verify-teardown] NO CONNECTION ({HOST!r})")
        return 1
    me = who.get("userName") or ""
    run_mode = bool(PREFIX) and not all_mode
    leaks: list[str] = []

    def gone(item, name, *creator_keys, schema_check=None) -> bool:
        """True if this resource should already be deleted (i.e. it's a leak)."""
        if run_mode:
            return _has_prefix(name) or bool(schema_check and schema_check())
        return _mine(item, me, *creator_keys)

    resp = _try("GET", "/api/2.0/serving-endpoints") or {}
    # Use only the LIST item's embedded served-entity config here (no per-endpoint
    # GET) — verify is advisory and must stay cheap regardless of endpoint count.
    def _ep_refs(e):
        ents = (e.get("config") or {}).get("served_entities") or []
        return any(_refs_run_schema(x.get("entity_name")) for x in ents)

    leaks += [
        f"serving-endpoint {e['name']!r}"
        for e in (resp.get("endpoints") or [])
        if e.get("name")
        and not _is_model_endpoint(e)
        and gone(e, e["name"], "creator", schema_check=lambda e=e: _ep_refs(e))
    ]
    resp = _try("GET", "/api/2.0/vector-search/endpoints") or {}
    leaks += [
        f"vector-search-endpoint {e['name']!r}"
        for e in (resp.get("endpoints") or [])
        if e.get("name") and gone(e, e["name"], "creator")
    ]
    resp = _try("GET", "/api/2.0/clusters/list") or {}
    leaks += [
        f"cluster {c['cluster_id']}"
        for c in (resp.get("clusters") or [])
        if c.get("cluster_source") != "JOB"
        and c.get("cluster_id")
        and gone(c, c.get("cluster_name"), "creator_user_name")
    ]
    resp = _try("GET", "/api/2.0/sql/warehouses") or {}
    leaks += [
        f"warehouse {w['id']}"
        for w in (resp.get("warehouses") or [])
        if w.get("id")
        and w.get("name") not in KEEP_WAREHOUSES
        and gone(w, w.get("name"), "creator_name")
    ]
    resp = _try("GET", "/api/2.1/jobs/list", query={"limit": "100"}) or {}
    leaks += [
        f"job {j['job_id']}"
        for j in (resp.get("jobs") or [])
        if j.get("job_id") and gone(j, (j.get("settings") or {}).get("name"), "creator_user_name")
    ]
    # DLT pipelines are billed while running — main() deletes them, so verify
    # must check them too (else a leaked prefixed pipeline reports clean).
    resp = _try("GET", "/api/2.0/pipelines", query={"max_results": "100"}) or {}
    leaks += [
        f"dlt-pipeline {p['pipeline_id']}"
        for p in (resp.get("statuses") or [])
        if p.get("pipeline_id")
        and gone(
            p,
            p.get("name"),
            "creator_user_name",
            schema_check=lambda pid=p["pipeline_id"]: _pipeline_refs_run_schema(pid),
        )
    ]
    resp = _try("GET", "/api/2.0/database/instances") or {}
    leaks += [
        f"lakebase-instance {i['name']!r}"
        for i in (resp.get("database_instances") or [])
        if i.get("name")
        and i["name"] not in KEEP_DB_INSTANCES
        and (_has_prefix(i["name"]) if run_mode else (i.get("creator") == me and bool(me)))
    ]

    if run_mode:
        # The run schema (and any prefixed catalog/schema) must be gone.
        if SCHEMA_FQN and _try("GET", f"/api/2.1/unity-catalog/schemas/{SCHEMA_FQN}") is not None:
            leaks.append(f"schema {SCHEMA_FQN!r}")
        cats = (_try("GET", "/api/2.1/unity-catalog/catalogs") or {}).get("catalogs") or []
        leaks += [
            f"catalog {c['name']!r}"
            for c in cats
            if c.get("name") and c["name"] not in KEEP_CATALOGS and _has_prefix(c["name"])
        ]
    else:
        cats = (_try("GET", "/api/2.1/unity-catalog/catalogs") or {}).get("catalogs") or []
        for cat in cats:
            name = cat.get("name")
            if not name or name in SKIP_SWEEP_CATALOGS:
                continue
            if name not in KEEP_CATALOGS and _mine(cat, me, "created_by"):
                leaks.append(f"catalog {name!r}")
                continue
            schemas = (
                _try("GET", "/api/2.1/unity-catalog/schemas", query={"catalog_name": name}) or {}
            ).get("schemas") or []
            leaks += [
                f"schema {sch.get('full_name') or (name + '.' + sch.get('name'))!r}"
                for sch in schemas
                if sch.get("name")
                and sch["name"] not in KEEP_SCHEMAS
                and _mine(sch, me, "created_by")
            ]

    if leaks:
        print("[databricks verify-teardown] LEAKS remain:\n  - " + "\n  - ".join(leaks))
        return 1
    print("[databricks verify-teardown] OK: connected; no agent resources remain")
    return 0


if __name__ == "__main__":
    import sys

    argv = sys.argv[1:]
    all_mode = "--all" in argv
    do_verify = "verify" in argv
    sys.exit(verify(all_mode) if do_verify else (main(all_mode) or 0))
