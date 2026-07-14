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
(ApiKey / JWT from HOPSWORKS_API_KEY + HOPSWORKS_HOST) is handled by the SDK and
the create call is `Connection.create_project`. We must connect WITHOUT selecting
a project — the project does not exist yet, creating it is this script's job — so
we use `hopsworks.connection()` rather than `hopsworks.login()`. login() cannot
do this: with no project to select it falls through to an interactive "Multiple
projects found …" prompt as soon as >1 project exists on the cluster (orphans
from interrupted earlier runs — the per-run teardown is scoped and sweeps only
its OWN project), and that prompt raises EOFError in our non-interactive
subprocess. `connection()` connects from the same env vars but never prompts.
See `_connect()`.

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


def _connect():
    """Connect to the cluster WITHOUT selecting a project.

    We deliberately do NOT use `hopsworks.login()` here: with no project to
    select, login() falls through to an interactive "Multiple projects found …"
    prompt the moment more than one project exists on the cluster (e.g. orphan
    projects left by interrupted earlier runs — the per-run teardown is scoped
    and only sweeps its OWN project). In our non-interactive serve subprocess
    that prompt raises EOFError, which used to be swallowed as "login skipped"
    so the run's project was never created and verify() then failed. The
    `hopsworks.connection()` factory connects from the same env vars but never
    prompts. We mirror login()'s env reads so behavior matches it otherwise.
    """
    import hopsworks

    hostname_verification = os.getenv("HOPSWORKS_HOSTNAME_VERIFICATION", "False").lower() in (
        "true",
        "1",
        "y",
        "yes",
    )
    return hopsworks.connection(
        host=os.environ.get("HOPSWORKS_HOST"),
        port=int(os.environ.get("HOPSWORKS_PORT", "443")),
        api_key_value=os.environ.get("HOPSWORKS_API_KEY"),
        hostname_verification=hostname_verification,
    )


def main() -> None:
    try:
        import hopsworks  # noqa: F401
    except Exception as e:  # SDK not importable in this venv → nothing to do
        print(f"[hopsworks setup] SDK unavailable: {e}")
        return

    try:
        conn = _connect()
    except Exception as e:
        # No reachable cluster → can't create a project; leave it to the agent.
        print(f"[hopsworks setup] connect skipped: {e}")
        return

    deadline = time.monotonic() + CREATE_RETRY_SECONDS
    attempt = 0
    while True:
        attempt += 1
        try:
            conn.create_project(PROJECT_NAME)
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
    """Twofold setup check (`setup.py verify`): (1) the cluster CONNECTS, then
    (2) THIS run's project (PROJECT_NAME, kept in HOPSWORKS_PROJECT) actually
    exists for the agent's later `hopsworks.login()` to auto-select. Both are
    hard gates — a missing project means setup's create silently failed and the
    run would not be valid. Read-only."""
    try:
        import hopsworks  # noqa: F401
    except Exception as e:
        print(f"[hopsworks verify-setup] SDK unavailable: {e}")
        return 1
    try:
        conn = _connect()
    except Exception as e:
        print(f"[hopsworks verify-setup] NO CONNECTION: {e}")
        return 1
    try:
        exists = conn.project_exists(PROJECT_NAME)
    except Exception as e:
        print(f"[hopsworks verify-setup] connected, but project check failed: {e}")
        return 1
    if not exists:
        print(
            f"[hopsworks verify-setup] connected, but project {PROJECT_NAME!r} "
            f"is missing (setup create did not succeed)"
        )
        return 1
    print(f"[hopsworks verify-setup] OK: connected; project {PROJECT_NAME!r} present")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(verify() if sys.argv[1:2] == ["verify"] else (main() or 0))
