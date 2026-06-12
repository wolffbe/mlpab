"""Databricks platform teardown — delete everything the agent created.

Run automatically by `banter run` (the interface `teardown:` step) at the START
and END of every challenge, so each run — and therefore each autoresearch
version — starts and ends with a clean workspace (same contract as the
hopsworks teardown).

Unlike Hopsworks there is NO project container whose deletion cascades: a
Databricks workspace holds many independent resource types, so cleanup is a
PER-TYPE sweep. Where the API reports a creator we only delete resources
CREATED BY the token's own user (the same ownership guard as the hopsworks
script); resources without creator metadata (workspace files, DBFS, MLflow
experiments) are scoped to the user's own paths or — for Unity Catalog default
schemas — assumed testbed-owned, since the workspace behind the .env token is
dedicated to the testbed.

Sweeps in COST order — the runner caps each aux step (~60 s), so if the script
is cut short the billable resources are already gone:
  1. model-serving + vector-search endpoints (billed while they exist)
  2. cancel active job runs, permanently delete clusters, SQL warehouses,
     DLT pipelines (billed while running)
  3. jobs, MLflow experiments + workspace-registry models
  4. Unity Catalog: own catalogs (force-cascades), own schemas in kept
     catalogs, own tables/volumes/functions/models in kept schemas — the
     `workspace` catalog and `default` schemas survive (the MCP server binds
     `-s workspace.default`)
  5. workspace home contents (/Users/<me>/…) and DBFS /FileStore contents

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
TOKEN = os.environ.get("DATABRICKS_TOKEN") or ""

# Never deleted, even when owned by the token user (default/system plumbing the
# next run relies on — the MCP interface binds the `workspace.default` schema).
KEEP_CATALOGS = {"system", "samples", "main", "workspace", "hive_metastore",
                 "__databricks_internal"}
# Catalogs not even swept inside (system-managed or not UC-governed).
SKIP_SWEEP_CATALOGS = {"system", "samples", "hive_metastore", "__databricks_internal"}
KEEP_SCHEMAS = {"default", "information_schema"}


def _api(method: str, path: str, payload: dict | None = None,
         query: dict | None = None) -> dict:
    url = HOST + path + (("?" + urllib.parse.urlencode(query)) if query else "")
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json",
    })
    with urllib.request.urlopen(req, timeout=20) as resp:
        body = resp.read()
    return json.loads(body) if body else {}


def _try(method: str, path: str, payload: dict | None = None,
         query: dict | None = None, label: str = "") -> dict | None:
    """Best-effort _api: None on any failure (logged only when labelled)."""
    try:
        return _api(method, path, payload, query)
    except Exception as e:
        if label:
            print(f"[databricks teardown] {label} failed: {e}")
        return None


def _mine(item: dict, me: str, *creator_keys: str) -> bool:
    """True unless the item provably belongs to someone else (hopsworks rule)."""
    for key in creator_keys:
        creator = item.get(key)
        if creator and me and creator != me:
            return False
    return True


def _delete_uc_model(full_name: str) -> None:
    quoted = urllib.parse.quote(full_name, safe="")
    versions = _try("GET", f"/api/2.1/unity-catalog/models/{quoted}/versions") or {}
    for v in versions.get("model_versions") or []:
        _try("DELETE", f"/api/2.1/unity-catalog/models/{quoted}/versions/{v.get('version')}")
    _try("DELETE", f"/api/2.1/unity-catalog/models/{quoted}",
         label=f"delete uc model {full_name!r}")
    print(f"[databricks teardown] deleted uc model {full_name!r}")


def _sweep_schema_objects(catalog: str, schema: str, me: str) -> None:
    """Delete the agent's tables / volumes / functions / models inside a kept schema."""
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
            if _try("DELETE", f"{base}/{quoted}", query=query,
                    label=f"delete uc {kind} {full!r}") is not None:
                print(f"[databricks teardown] deleted uc {kind} {full!r}")
    resp = _try("GET", "/api/2.1/unity-catalog/models", query=scope) or {}
    for obj in resp.get("registered_models") or []:
        full = obj.get("full_name")
        if full and _mine(obj, me, "created_by", "owner"):
            _delete_uc_model(full)


def _sweep_unity_catalog(me: str) -> None:
    catalogs = (_try("GET", "/api/2.1/unity-catalog/catalogs") or {}).get("catalogs") or []
    for cat in catalogs:
        name = cat.get("name")
        if not name or name in SKIP_SWEEP_CATALOGS:
            continue
        # Agent-created catalog → force delete cascades all of its contents.
        if name not in KEEP_CATALOGS and _mine(cat, me, "created_by"):
            if _try("DELETE", f"/api/2.1/unity-catalog/catalogs/{name}",
                    query={"force": "true"}, label=f"delete catalog {name!r}") is not None:
                print(f"[databricks teardown] deleted catalog {name!r}")
            continue
        # Kept catalog → sweep agent-created schemas, then objects in kept schemas.
        schemas = (_try("GET", "/api/2.1/unity-catalog/schemas",
                        query={"catalog_name": name}) or {}).get("schemas") or []
        for sch in schemas:
            sname = sch.get("name")
            if not sname:
                continue
            if sname not in KEEP_SCHEMAS and _mine(sch, me, "created_by"):
                full = sch.get("full_name") or f"{name}.{sname}"
                if _try("DELETE", f"/api/2.1/unity-catalog/schemas/{full}",
                        query={"force": "true"}, label=f"delete schema {full!r}") is not None:
                    print(f"[databricks teardown] deleted schema {full!r}")
            elif sname != "information_schema":
                _sweep_schema_objects(name, sname, me)


def main() -> None:
    if not HOST or not TOKEN:
        print("[databricks teardown] DATABRICKS_HOST/DATABRICKS_TOKEN unset — nothing to do")
        return

    who = _try("GET", "/api/2.0/preview/scim/v2/Me", label="identify user")
    if not who:  # workspace unreachable / bad token → nothing we can clean
        return
    me = who.get("userName") or ""

    # 1) Billed-while-they-exist endpoints.
    resp = _try("GET", "/api/2.0/serving-endpoints", label="list serving endpoints") or {}
    for ep in resp.get("endpoints") or []:
        if ep.get("name") and _mine(ep, me, "creator"):
            if _try("DELETE", f"/api/2.0/serving-endpoints/{ep['name']}",
                    label=f"delete serving endpoint {ep['name']!r}") is not None:
                print(f"[databricks teardown] deleted serving endpoint {ep['name']!r}")
    resp = _try("GET", "/api/2.0/vector-search/endpoints") or {}
    for ep in resp.get("endpoints") or []:
        if ep.get("name") and _mine(ep, me, "creator"):
            if _try("DELETE", f"/api/2.0/vector-search/endpoints/{ep['name']}") is not None:
                print(f"[databricks teardown] deleted vector-search endpoint {ep['name']!r}")

    # 2) Billed-while-running compute.
    token = None
    while True:
        query = {"active_only": "true"}
        if token:
            query["page_token"] = token
        resp = _try("GET", "/api/2.1/jobs/runs/list", query=query) or {}
        for run in resp.get("runs") or []:
            if run.get("run_id") and _mine(run, me, "creator_user_name"):
                _try("POST", "/api/2.1/jobs/runs/cancel", {"run_id": run["run_id"]})
                print(f"[databricks teardown] cancelled run {run['run_id']}")
        token = resp.get("next_page_token")
        if not token:
            break

    resp = _try("GET", "/api/2.0/clusters/list", label="list clusters") or {}
    for cl in resp.get("clusters") or []:
        if cl.get("cluster_source") == "JOB":  # ephemeral; dies with its run
            continue
        if cl.get("cluster_id") and _mine(cl, me, "creator_user_name"):
            if _try("POST", "/api/2.0/clusters/permanent-delete",
                    {"cluster_id": cl["cluster_id"]},
                    label=f"delete cluster {cl['cluster_id']}") is not None:
                print(f"[databricks teardown] deleted cluster {cl['cluster_id']} "
                      f"({cl.get('cluster_name')!r})")

    resp = _try("GET", "/api/2.0/sql/warehouses") or {}
    for wh in resp.get("warehouses") or []:
        if wh.get("id") and _mine(wh, me, "creator_name"):
            if _try("DELETE", f"/api/2.0/sql/warehouses/{wh['id']}") is not None:
                print(f"[databricks teardown] deleted sql warehouse {wh['id']} "
                      f"({wh.get('name')!r})")

    token = None
    while True:
        query = {"max_results": "100"}
        if token:
            query["page_token"] = token
        resp = _try("GET", "/api/2.0/pipelines", query=query) or {}
        for pl in resp.get("statuses") or []:
            if pl.get("pipeline_id") and _mine(pl, me, "creator_user_name"):
                if _try("DELETE", f"/api/2.0/pipelines/{pl['pipeline_id']}") is not None:
                    print(f"[databricks teardown] deleted dlt pipeline {pl['pipeline_id']}")
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
            if job.get("job_id") and _mine(job, me, "creator_user_name"):
                if _try("POST", "/api/2.1/jobs/delete", {"job_id": job["job_id"]}) is not None:
                    print(f"[databricks teardown] deleted job {job['job_id']}")
        token = resp.get("next_page_token")
        if not token:
            break

    home = f"/Users/{me}"
    resp = _try("POST", "/api/2.0/mlflow/experiments/search", {"max_results": 1000}) or {}
    for exp in resp.get("experiments") or []:
        # Experiments carry no creator field; the user's home path scopes them.
        if (exp.get("name") or "").startswith(home + "/") and exp.get("experiment_id"):
            if _try("POST", "/api/2.0/mlflow/experiments/delete",
                    {"experiment_id": exp["experiment_id"]}) is not None:
                print(f"[databricks teardown] deleted experiment {exp['name']!r}")

    resp = _try("GET", "/api/2.0/mlflow/registered-models/search",
                query={"max_results": "1000"}) or {}
    for rm in resp.get("registered_models") or []:
        if rm.get("name") and _mine(rm, me, "user_id"):
            if _try("DELETE", "/api/2.0/mlflow/registered-models/delete",
                    {"name": rm["name"]}) is not None:
                print(f"[databricks teardown] deleted registered model {rm['name']!r}")

    # 4) Unity Catalog (absent on non-UC workspaces — every call is best-effort).
    _sweep_unity_catalog(me)

    # 5) Files: workspace home contents + DBFS /FileStore contents (the dirs
    #    themselves stay — they are default plumbing).
    resp = _try("GET", "/api/2.0/workspace/list", query={"path": home}) or {}
    for obj in resp.get("objects") or []:
        path = obj.get("path")
        if path and _try("POST", "/api/2.0/workspace/delete",
                         {"path": path, "recursive": True},
                         label=f"delete workspace path {path!r}") is not None:
            print(f"[databricks teardown] deleted workspace path {path!r}")

    resp = _try("GET", "/api/2.0/dbfs/list", query={"path": "/FileStore"}) or {}
    for f in resp.get("files") or []:
        path = f.get("path")
        if path and _try("POST", "/api/2.0/dbfs/delete",
                         {"path": path, "recursive": True}) is not None:
            print(f"[databricks teardown] deleted dbfs path {path!r}")

    print("[databricks teardown] done")


if __name__ == "__main__":
    main()
