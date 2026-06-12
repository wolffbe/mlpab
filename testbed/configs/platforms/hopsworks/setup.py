"""Hopsworks platform setup — create one empty project for the engineer to work in.

Run automatically by `banter run` (the interface `serve:` step) at the START of
every challenge, right AFTER `teardown.py` has wiped the cluster clean — so each
run (and therefore each autoresearch version) begins with exactly ONE fresh,
empty project and nothing leaked from an earlier run. The matching removal is
`teardown.py`, which deletes every project the API key owns at both the start and
the end of every run; the empty project this script creates is cleaned up there
(no separate delete needed here).

Why pre-create it instead of letting the agent do it? Creating a project is
platform plumbing, not part of the FTI lifecycle we measure (the whitelist starts
at feature-group create). Pre-creating exactly one project means the engineer's
`hopsworks.login()` lands in it automatically (a single project is selected
without prompting), so the agent spends its effort on features/models — and a
flaky project-create never confounds a run. Created for EVERY interface (cli /
sdk / mcp), since each interface is its own run.

We reuse the SDK's authenticated client the same way `teardown.py` does: auth
(ApiKey / JWT from HOPSWORKS_API_KEY + HOPSWORKS_HOST) and the create call are
handled by `hopsworks.create_project`. `hopsworks.login()` does NOT require an
existing project on a self-hosted cluster — it returns no active project and
leaves the connection live, which is all we need to create one.

Best-effort by design: invoked via the interface `serve:` step, whose runner
(`_run_aux`) ignores failures and discards output. Nothing here may raise out of
`main()` — a setup hiccup must never fail an engineer run.

The project name is UNIQUE PER RUN (banter<6 hex chars>): the backend deletes a
project's Kubernetes namespace and Kafka topics/ACLs ASYNCHRONOUSLY, so an
immediate same-name re-create either fails outright (HTTP 500 / errorCode
150051 "Namespace:banter is being deleted") or — worse — comes up colliding
with the old namesake's still-being-cleaned Kafka state, and every feature
group insert dies server-side with KafkaError TOPIC_AUTHORIZATION_FAILED
(observed live 2026-06-12). A fresh name sidesteps both races. Uniqueness is
safe: `teardown.py` deletes every project the key owns regardless of name, and
the agent's `hopsworks.login()` auto-selects the single existing project. The
create still RETRIES briefly for transient backend hiccups; the runner's serve
timeout (runner.py `_run_aux`) must stay comfortably above
CREATE_RETRY_SECONDS.
"""
from __future__ import annotations

import secrets
import time

# Unique-per-run, alphanumeric project name (see module docstring). A single
# existing project means the agent's login selects it with no prompt.
PROJECT_NAME = f"banter{secrets.token_hex(3)}"

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
            print(
                f"[hopsworks setup] created empty project {PROJECT_NAME!r}"
                f" (attempt {attempt})"
            )
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
            print(
                f"[hopsworks setup] create attempt {attempt} failed, retrying: {e}"
            )
            time.sleep(CREATE_RETRY_SLEEP)


if __name__ == "__main__":
    main()
