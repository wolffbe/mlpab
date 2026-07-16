#!/usr/bin/env bash
# One-time Hopsworks bootstrap for the mlpab testbed: collect the cluster
# endpoint + API key into testbed/.env. Idempotent — safe to re-run.
#
# Hopsworks has no resources to provision from here (the per-run setup.py
# pre-creates the project container). All this step guarantees is that the two
# credentials every Hopsworks interface (sdk/cli/mcp) reads from the environment
# are present and non-empty:
#
#   * HOPSWORKS_HOST    — the cluster endpoint (e.g. https://my-cluster.hopsworks.ai)
#   * HOPSWORKS_API_KEY — an API key for that cluster (Account Settings → API keys)
#
# Both are REQUIRED; this script refuses to finish until both are set.
#
#   bash configs/platforms/hopsworks/bootstrap.sh     (or: make setup → hopsworks)
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

say "Hopsworks credentials (written to $ENV_FILE)"
HOPSWORKS_HOST="$(prompt HOPSWORKS_HOST 'HOPSWORKS_HOST (e.g. https://my-cluster.hopsworks.ai)')"
HOPSWORKS_API_KEY="$(prompt HOPSWORKS_API_KEY 'HOPSWORKS_API_KEY' secret)"

[ -n "$HOPSWORKS_HOST" ]    || die "HOPSWORKS_HOST is required."
[ -n "$HOPSWORKS_API_KEY" ] || die "HOPSWORKS_API_KEY is required."

set_env HOPSWORKS_HOST "$HOPSWORKS_HOST"
set_env HOPSWORKS_API_KEY "$HOPSWORKS_API_KEY"
say "wrote HOPSWORKS_HOST / HOPSWORKS_API_KEY"

say "DONE. Validate with:  mlpab check configs/treatments/hopsworks/<config>.yaml"
say "      (or a single interface:  mlpab test configs/platforms/hopsworks/sdk.yaml)"
