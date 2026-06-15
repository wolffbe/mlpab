"""Hopsworks platform setup — create one empty project for the engineer to work in.

Run automatically by `mlpab run` (the interface `serve:` step) at the START of
every challenge, right AFTER `teardown.py` has swept the run's own project — so
each run (and therefore each autoresearch version) begins with one fresh, empty
project named HOPSWORKS_PROJECT and nothing leaked from an earlier run of the
same name. The matching removal is `teardown.py`, which deletes exactly the
HOPSWORKS_PROJECT project at both the start and the end of every run; the empty
project this script creates is cleaned up there (no separate delete needed here).

Why pre-create it instead of letting the agent do it? Creating a project is
platform plumbing, not part of the FTI lifecycle we measure (the whitelist starts
at feature-group create). Pre-creating the run's named project means the
engineer's `hopsworks.login()` lands in it automatically (HOPSWORKS_PROJECT
selects it without prompting, even when other parallel runs' projects coexist on
the cluster), so the agent spends its effort on features/models — and a flaky
project-create never confounds a run. Created for EVERY interface (cli /
sdk / mcp), since each interface is its own run.

We reuse the SDK's authenticated client the same way `teardown.py` does: auth
(ApiKey / JWT from HOPSWORKS_API_KEY + HOPSWORKS_HOST) and the create call are
handled by `hopsworks.create_project`. The login here must connect WITHOUT
selecting a project: the runner exports HOPSWORKS_PROJECT (the per-run name)
before this step, and `hopsworks.login()` reads that env var and tries to SELECT
the named project — which does not exist yet, since creating it is this script's
job. So `main()` pops HOPSWORKS_PROJECT before login (the name is already
captured in PROJECT_NAME) and recreates the project by name.

Best-effort by design: invoked via the interface `serve:` step, whose runner
(`_run_aux`) ignores failures and discards output. Nothing here may raise out of
`main()` — a setup hiccup must never fail an engineer run.

The project name is UNIQUE PER RUN (mlpab<6 hex chars>): the backend deletes a
project's Kubernetes namespace and Kafka topics/ACLs ASYNCHRONOUSLY, so an
immediate same-name re-create either fails outright (HTTP 500 / errorCode
150051 "Namespace:mlpab is being deleted") or — worse — comes up colliding
with the old namesake's still-being-cleaned Kafka state, and every feature
group insert dies server-side with KafkaError TOPIC_AUTHORIZATION_FAILED
(observed live 2026-06-12). A fresh name sidesteps both races. Uniqueness is
safe: the runner mints HOPSWORKS_PROJECT fresh per run, `teardown.py` deletes
exactly that project, and the agent's `hopsworks.login()` selects it by name via
the same env var. The create still RETRIES briefly for transient backend
hiccups; the runner's serve timeout (runner.py `_run_aux`) must stay comfortably
above CREATE_RETRY_SECONDS.
"""

from __future__ import annotations

import os
import secrets
import time

# Unique-per-run, alphanumeric project name (see module docstring). The runner
# generates it once per run and exports it as HOPSWORKS_PROJECT so setup
# (creates it), the agent (login() auto-selects it), the grader's adapter (reads
# back through it), and teardown (deletes only it) all agree on the SAME name —
# which is what lets two hopsworks runs share a cluster. We honor that env value
# and only mint our own when invoked standalone (no runner), so the agent's
# bare login() and the per-run teardown still line up on the name we create.
PROJECT_NAME = os.environ.get("HOPSWORKS_PROJECT") or f"mlpab{secrets.token_hex(3)}"

# How long to keep retrying the create while the backend finishes tearing down
# the previous run's namespace. A clean create takes ~20s; namespace finalizers
# usually clear within seconds but can lag under load.
CREATE_RETRY_SECONDS = 180
CREATE_RETRY_SLEEP = 5


def main() -> None:
    try:
        import hopsworks
    except Exception as e:  # SDK not importable in this venv → nothing to do
        print(f"[hopsworks setup] SDK unavailable: {e}")
        return

    # Connect WITHOUT selecting a project. The runner exports HOPSWORKS_PROJECT
    # (the per-run name) before this step, and `hopsworks.login()` reads that env
    # var and tries to SELECT it — but the project does not exist yet, that is
    # exactly what this script creates. Selecting it would raise ("Could not find
    # project …") and abort the create. PROJECT_NAME has already captured the
    # name at module load, so dropping the env var here is safe; we recreate it
    # below by name. (verify() deliberately keeps the env var set — by then the
    # project exists and the check mirrors the agent's later bare login().)
    os.environ.pop("HOPSWORKS_PROJECT", None)
    try:
        hopsworks.login()  # reads HOPSWORKS_API_KEY / HOPSWORKS_HOST from env
    except Exception as e:
        # No reachable cluster → can't create a project; leave it to the agent.
        print(f"[hopsworks setup] login skipped: {e}")
        return

    deadline = time.monotonic() + CREATE_RETRY_SECONDS
    attempt = 0
    while True:
        attempt += 1
        try:
            hopsworks.create_project(PROJECT_NAME)
            print(f"[hopsworks setup] created empty project {PROJECT_NAME!r} (attempt {attempt})")
            return
        except Exception as e:
            # Most commonly the previous run's namespace is still being deleted
            # (errorCode 150051) — transient by the backend's own admission, so
            # retry until the deadline. Anything still failing then is a real
            # backend problem; stay best-effort and let the run proceed.
            if time.monotonic() >= deadline:
                print(
                    f"[hopsworks setup] create of project {PROJECT_NAME!r} gave up"
                    f" after {attempt} attempt(s): {e}"
                )
                return
            print(f"[hopsworks setup] create attempt {attempt} failed, retrying: {e}")
            time.sleep(CREATE_RETRY_SLEEP)


def verify() -> int:
    """Twofold setup check (`setup.py verify`): (1) the cluster CONNECTS (login),
    then (2) a project is AVAILABLE for the agent's login to select. Connection
    is the hard gate; the project count is best-effort. Read-only."""
    try:
        import hopsworks
    except Exception as e:
        print(f"[hopsworks verify-setup] SDK unavailable: {e}")
        return 1
    try:
        hopsworks.login()  # reads HOPSWORKS_API_KEY / HOPSWORKS_HOST from env
    except Exception as e:
        print(f"[hopsworks verify-setup] NO CONNECTION (or no project): {e}")
        return 1
    try:
        from hopsworks_common import client

        teams = client.get_instance()._send_request("GET", ["project"]) or []
        print(f"[hopsworks verify-setup] OK: connected; {len(teams)} project(s) available")
    except Exception as e:
        print(
            f"[hopsworks verify-setup] OK: connected; project list unavailable (best-effort): {e}"
        )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(verify() if sys.argv[1:2] == ["verify"] else (main() or 0))
