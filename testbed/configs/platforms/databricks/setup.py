"""Databricks platform setup — ensure the `workspace.default` UC schema exists.

Run automatically by `mlpab run` (the interface `serve:` step) at the START of
every challenge, right AFTER `teardown.py` has swept the workspace — same
contract as the hopsworks setup.

Unlike Hopsworks there is NO project container to pre-create: the token scopes
to the whole workspace and every interface authenticates without any
pre-existing resource. The one piece of platform plumbing worth guaranteeing is
the Unity Catalog schema `workspace.default` — the MCP interface launches its
server bound to it (`unitycatalog-mcp -s workspace.default`, see mcp.yaml), and
it is the natural landing zone for feature tables on the other interfaces.
`teardown.py` deliberately never deletes the `workspace` catalog or `default`
schemas, so this is normally a no-op guard that only acts if the schema went
missing (e.g. a manually wiped workspace).

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

CATALOG = "workspace"
SCHEMA = "default"


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


def main() -> None:
    if not HOST or not TOKEN:
        print("[databricks setup] DATABRICKS_HOST/DATABRICKS_TOKEN unset — nothing to do")
        return

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


def verify() -> int:
    """Twofold setup check (`setup.py verify`): (1) the workspace CONNECTS with
    the run's token, then (2) the `workspace.default` schema setup guarantees is
    PRESENT. Exit 0 iff ready, non-zero with a reason otherwise — the runner
    fails the run on non-zero, so the agent never works against a platform it
    can't reach. Read-only.

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
