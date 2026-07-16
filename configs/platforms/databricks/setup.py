"""Databricks platform setup — ensure this run's UC landing-zone schema exists.

Run automatically by `mlpab run` (the interface `serve:` step) at the START of
every challenge, right AFTER `teardown.py` has swept the workspace — same
contract as the hopsworks setup.

Unlike Hopsworks there is NO project container to pre-create: the token scopes
to the whole workspace and every interface authenticates without any
pre-existing resource. Instead the runner mints a PER-RUN Unity Catalog schema
name in the `workspace` catalog (`MLPAB_DATABRICKS_SCHEMA=workspace.mlpab<hex>`,
see src/mlpab/runner.py) that is this run's landing zone for feature tables and
every other UC object — the single source of truth setup creates, the agent
lands tables in, the grader reads back through, and teardown force-deletes ONLY.
Scoping the schema per run is what lets two databricks runs share one workspace
token without their teardowns deleting each other's tables. When the env var is
unset (a manual/single-run invocation) we fall back to `workspace.default`.

Talks to the REST API with the stdlib only (urllib), like teardown.py, so the
same script works for cli, sdk, and mcp runs.

Best-effort by design: invoked via the interface `serve:` step, whose runner
(`_run_aux`) ignores failures and discards output. Nothing here may raise out of
`main()` — a setup hiccup (e.g. a workspace without Unity Catalog) must never
fail an engineer run.
"""

from __future__ import annotations

import json
import os
import urllib.request

HOST = (os.environ.get("DATABRICKS_HOST") or "").rstrip("/")
# DATABRICKS_HOST is commonly set scheme-less (e.g. dbc-xxxx.cloud.databricks.com);
# the CLI/SDK tolerate that but urllib needs an explicit scheme, so add one.
if HOST and "://" not in HOST:
    HOST = "https://" + HOST
TOKEN = os.environ.get("DATABRICKS_TOKEN") or ""

# Per-run landing-zone schema as `<catalog>.<schema>` (runner sets it; falls
# back to the conventional workspace.default for manual/single-run use).
_SCHEMA_FQN = os.environ.get("MLPAB_DATABRICKS_SCHEMA") or "workspace.default"
CATALOG, SCHEMA = (_SCHEMA_FQN.split(".", 1) + ["default"])[:2]
# Stable SQL warehouse the GRADER reads through (see evals/adapters/databricks.py).
# teardown.py preserves it by name so it survives the per-run sweep.
GRADER_WAREHOUSE = "mlpab-grader"


def _api(method: str, path: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        HOST + path,
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


def _ensure_schema() -> None:
    full = f"{CATALOG}.{SCHEMA}"
    try:
        _api("GET", f"/api/2.1/unity-catalog/schemas/{full}")
        print(f"[databricks setup] schema {full!r} already exists")
        return
    except Exception:
        pass  # missing (or no UC at all) → try to create it
    try:
        _api("POST", "/api/2.1/unity-catalog/schemas", {"name": SCHEMA, "catalog_name": CATALOG})
        print(f"[databricks setup] created schema {full!r}")
    except Exception as e:
        # Workspace without Unity Catalog, or missing catalog — the agent can
        # still work elsewhere, so don't fail the run.
        print(f"[databricks setup] create of schema {full!r} skipped: {e}")


def _ensure_grader_warehouse() -> None:
    """Provision a small, stable SQL warehouse the GRADER reads through.

    Readback executes SQL on a warehouse. Relying on whatever warehouse happens
    to exist makes grading flaky: a parallel run's teardown can sweep it out
    from under a readback ("no SQL warehouse available"), and a cold classic
    warehouse can leave a statement PENDING past the wait cap. A named,
    auto-stopping warehouse that teardown.py preserves gives every readback a
    stable target; serverless (preferred) also starts in seconds. Best-effort:
    never raises — a workspace without SQL warehouses still grades via whatever
    the adapter can find."""
    try:
        resp = _api("GET", "/api/2.0/sql/warehouses")
        for wh in resp.get("warehouses") or []:
            if wh.get("name") == GRADER_WAREHOUSE:
                print(
                    f"[databricks setup] grader warehouse {GRADER_WAREHOUSE!r} "
                    f"present ({wh.get('id')})"
                )
                return
    except Exception as e:
        print(f"[databricks setup] grader warehouse lookup skipped: {e}")
        return

    base = {
        "name": GRADER_WAREHOUSE,
        "cluster_size": "2X-Small",
        "min_num_clusters": 1,
        "max_num_clusters": 1,
        "auto_stop_mins": 10,
    }
    # Prefer serverless (fast cold start); fall back to a classic PRO warehouse
    # where serverless is not enabled on the account.
    for payload in ({**base, "enable_serverless_compute": True, "warehouse_type": "PRO"}, base):
        try:
            created = _api("POST", "/api/2.0/sql/warehouses", payload)
            kind = "serverless" if payload.get("enable_serverless_compute") else "classic"
            print(
                f"[databricks setup] created {kind} grader warehouse "
                f"{GRADER_WAREHOUSE!r} ({created.get('id')})"
            )
            return
        except Exception as e:
            last = e
    print(f"[databricks setup] grader warehouse provision skipped: {last}")


def main() -> None:
    if not HOST or not TOKEN:
        print("[databricks setup] DATABRICKS_HOST/DATABRICKS_TOKEN unset — nothing to do")
        return
    _ensure_schema()
    _ensure_grader_warehouse()


def verify() -> int:
    """Twofold setup check (`setup.py verify`): (1) the workspace CONNECTS with
    the run's token, then (2) this run's landing-zone schema (CATALOG.SCHEMA,
    per-run when MLPAB_DATABRICKS_SCHEMA is set) is PRESENT. Exit 0 iff ready,
    non-zero with a reason otherwise — the runner fails the run on non-zero, so
    the agent never works against a platform it can't reach. Read-only.

    Connection is the hard gate; the schema is best-effort (setup.py itself only
    creates it where Unity Catalog exists), so a non-UC workspace that connects
    still passes — it's reported, not failed.
    """
    if not HOST or not TOKEN:
        print("[databricks verify-setup] DATABRICKS_HOST/DATABRICKS_TOKEN unset")
        return 1
    try:
        _api("GET", "/api/2.0/preview/scim/v2/Me")
    except Exception as e:
        print(f"[databricks verify-setup] NO CONNECTION ({HOST!r}): {e}")
        return 1
    full = f"{CATALOG}.{SCHEMA}"
    try:
        _api("GET", f"/api/2.1/unity-catalog/schemas/{full}")
        print(f"[databricks verify-setup] OK: connected; schema {full!r} present")
    except Exception as e:
        print(
            f"[databricks verify-setup] OK: connected; schema {full!r} absent "
            f"(no Unity Catalog?) — best-effort, not failing: {e}"
        )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(verify() if sys.argv[1:2] == ["verify"] else (main() or 0))
