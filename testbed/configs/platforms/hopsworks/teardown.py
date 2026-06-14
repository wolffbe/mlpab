"""Hopsworks platform teardown — delete every project the agent created.

Run automatically by `mlpab run` (the interface `teardown:` step) at the START
and END of every challenge, so each run — and therefore each autoresearch
version — starts and ends with a clean cluster. Without this, feature groups /
feature views / models from earlier runs leak into later ones (the exact bug
that gave SDK v0 a 0.22 score: a stale `cactus_fv` left 49 feature columns
zero-filled).

The cleanest reset is to delete the whole PROJECT, which cascades to all its
feature store + model registry + dataset contents. The backend endpoint is
`POST /hopsworks-api/api/project/{projectId}/delete`
(hopsworks-ee `ProjectService.removeProjectAndFiles`, DATA_OWNER only). We reuse
the SDK's authenticated REST client (`hopsworks_common.client`) so auth (ApiKey /
JWT from HOPSWORKS_API_KEY + HOPSWORKS_HOST) and token refresh are handled for us.

Best-effort by design: invoked via the interface `teardown:` step, whose runner
(`_run_aux`) ignores failures and discards output. Nothing here may raise out of
`main()` — a teardown hiccup must never fail an engineer run. It deletes every
project OWNED by the API key's user, so it cleans whatever the agent created
regardless of the project name it chose.
"""

from __future__ import annotations


def main() -> None:
    try:
        import hopsworks
        from hopsworks_common import client
    except Exception as e:  # SDK not importable in this venv → nothing to do
        print(f"[hopsworks teardown] SDK unavailable: {e}")
        return

    try:
        hopsworks.login()  # reads HOPSWORKS_API_KEY / HOPSWORKS_HOST from env
    except Exception as e:
        # No reachable cluster / no project to connect to → nothing to clean.
        print(f"[hopsworks teardown] login skipped: {e}")
        return

    try:
        c = client.get_instance()
        teams = c._send_request("GET", ["project"]) or []
    except Exception as e:
        print(f"[hopsworks teardown] could not list projects: {e}")
        return

    # Identify the current user so we only delete projects THIS key owns.
    my_uid = None
    for t in teams:
        my_uid = (t.get("user") or {}).get("uid")
        if my_uid is not None:
            break

    deleted = 0
    seen: set = set()
    for t in teams:
        proj = t.get("project") or {}
        pid = proj.get("id")
        owner_uid = (proj.get("owner") or {}).get("uid")
        if pid is None or pid in seen:
            continue
        if my_uid is not None and owner_uid is not None and owner_uid != my_uid:
            continue  # not ours — leave it alone
        seen.add(pid)
        try:
            c._send_request("POST", ["project", pid, "delete"])
            deleted += 1
            print(f"[hopsworks teardown] deleted project {proj.get('name')!r} (id={pid})")
        except Exception as e:
            print(f"[hopsworks teardown] delete of project id={pid} failed: {e}")

    print(f"[hopsworks teardown] done — {deleted} project(s) deleted")


def verify() -> int:
    """Twofold teardown check (`teardown.py verify`): confirm no project OWNED by
    this key survives the sweep. Read-only. Only positively reports a leak
    (return 1) — if it can't connect/list (e.g. post-teardown there's no project
    to attach to), it assumes clean (return 0); the next run's start-teardown
    re-sweeps regardless, and the runner only WARNS on this check."""
    try:
        import hopsworks
        from hopsworks_common import client
    except Exception as e:
        print(f"[hopsworks verify-teardown] SDK unavailable (assuming clean): {e}")
        return 0
    try:
        hopsworks.login()
        teams = client.get_instance()._send_request("GET", ["project"]) or []
    except Exception as e:
        print(f"[hopsworks verify-teardown] could not confirm (assuming clean): {e}")
        return 0
    my_uid = None
    for t in teams:
        my_uid = (t.get("user") or {}).get("uid")
        if my_uid is not None:
            break
    owned, seen = [], set()
    for t in teams:
        proj = t.get("project") or {}
        pid = proj.get("id")
        if pid is None or pid in seen:
            continue
        owner_uid = (proj.get("owner") or {}).get("uid")
        if my_uid is not None and owner_uid is not None and owner_uid != my_uid:
            continue
        seen.add(pid)
        owned.append(proj.get("name"))
    if owned:
        print("[hopsworks verify-teardown] LEAKS: project(s) " + ", ".join(map(str, owned)))
        return 1
    print("[hopsworks verify-teardown] OK: connected; no owned projects remain")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(verify() if sys.argv[1:2] == ["verify"] else (main() or 0))
