#!/usr/bin/env bash
# One-time Databricks bootstrap for the mlpab testbed: collect the workspace
# host + personal access token into testbed/.env. Idempotent — safe to re-run.
#
# Databricks has no resources to provision from here; all this step guarantees
# is that the two credentials every Databricks interface (sdk/cli) reads from
# the environment are present and non-empty:
#
#   * DATABRICKS_HOST  — the workspace URL (e.g. https://dbc-xxxx.cloud.databricks.com)
#   * DATABRICKS_TOKEN — a personal access token (Settings → Developer → Access tokens)
#
# Both are REQUIRED; this script refuses to finish until both are set.
#
#   bash configs/platforms/databricks/bootstrap.sh    (or: make setup → databricks)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTBED="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ENV_FILE="$TESTBED/.env"

say() { printf '\n>> %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

set_env() {  # set_env KEY VALUE — replace existing line or append; value-safe
  local key="$1"; shift; local val="$*" tmp
  tmp="$(mktemp)"; touch "$ENV_FILE"
  grep -v "^${key}=" "$ENV_FILE" > "$tmp" 2>/dev/null || true
  printf '%s=%s\n' "$key" "$val" >> "$tmp"
  mv "$tmp" "$ENV_FILE"
}

env_get() {
  local key="$1" val
  val="$(printenv "$key" 2>/dev/null || true)"
  if [ -z "$val" ] && [ -f "$ENV_FILE" ]; then
    val="$(grep "^${key}=" "$ENV_FILE" 2>/dev/null | tail -1 | cut -d= -f2-)"
  fi
  printf '%s' "$val"
}

prompt() {   # prompt VAR PROMPT [secret]; keeps the current value on empty input
  local var="$1" msg="$2" secret="${3:-}" cur input shown
  cur="$(env_get "$var")"
  shown="$cur"; [ -n "$secret" ] && [ -n "$cur" ] && shown="********"
  if [ -n "$cur" ]; then msg="$msg [$shown]"; fi
  if [ -n "$secret" ]; then read -rsp "  $msg: " input; echo; else read -rp "  $msg: " input; fi
  printf '%s' "${input:-$cur}"
}

say "Databricks credentials (written to $ENV_FILE)"
DATABRICKS_HOST="$(prompt DATABRICKS_HOST 'DATABRICKS_HOST (e.g. https://dbc-xxxx.cloud.databricks.com)')"
DATABRICKS_TOKEN="$(prompt DATABRICKS_TOKEN 'DATABRICKS_TOKEN' secret)"

[ -n "$DATABRICKS_HOST" ]  || die "DATABRICKS_HOST is required."
[ -n "$DATABRICKS_TOKEN" ] || die "DATABRICKS_TOKEN is required."

set_env DATABRICKS_HOST "$DATABRICKS_HOST"
set_env DATABRICKS_TOKEN "$DATABRICKS_TOKEN"
say "wrote DATABRICKS_HOST / DATABRICKS_TOKEN"

say "DONE. Validate with:  mlpab check configs/treatments/databricks/<config>.yaml"
say "      (or a single interface:  mlpab test configs/platforms/databricks/sdk.yaml)"
