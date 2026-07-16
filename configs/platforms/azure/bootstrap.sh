#!/usr/bin/env bash
# One-time Azure bootstrap for the mlpab testbed: install the CLI, log in, create
# the resource group + Azure ML workspace + service principal, grant roles, and
# write the AZURE_* keys into testbed/.env. Idempotent — safe to re-run.
#
# Encodes the real-world gotchas this subscription/tenant hit:
#   * Region policy ("Allowed resource deployment regions") restricts where you
#     can deploy — we read it and pick an allowed EU region automatically.
#   * The tenant may FORBID service-principal secrets ("Credential type not
#     allowed") — we fall back to a CERTIFICATE credential and use
#     AZURE_CLIENT_CERTIFICATE_PATH (DefaultAzureCredential / az --certificate).
#
#   bash configs/platforms/azure/bootstrap.sh
# Overridable via env: RG, WS, SP, LOCATION (preferred region; subject to policy).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTBED="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ENV_FILE="$TESTBED/.env"
CERT_DST="$TESTBED/.azure/azure-sp.pem"

RG="${RG:-mlpab-rg}"
WS="${WS:-mlpab-ws}"
SP="${SP:-mlpab-sp}"
PREF_LOCATION="${LOCATION:-westeurope}"   # preferred; overridden if policy disallows
# EU-first preference order used when the region policy restricts us.
REGION_PREF=(westeurope northeurope swedencentral francecentral germanywestcentral \
             ***REDACTED*** spaincentral italynorth austriaeast norwayeast)

say() { printf '\n>> %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

set_env() {  # set_env KEY VALUE — replace existing line or append; value-safe
  local key="$1"; shift; local val="$*" tmp
  val="${val//$'\r'/}"; val="${val//$'\n'/}"   # never split a value across lines
  tmp="$(mktemp)"; touch "$ENV_FILE"
  grep -v "^${key}=" "$ENV_FILE" 2>/dev/null | cat -s > "$tmp" || true
  printf '%s=%s\n' "$key" "$val" >> "$tmp"
  mv "$tmp" "$ENV_FILE"
}

env_sep() {  # one blank-line separator before this bootstrap's block (idempotent)
  touch "$ENV_FILE"
  [ -s "$ENV_FILE" ] && [ -n "$(tail -c1 "$ENV_FILE")" ] && printf '\n' >> "$ENV_FILE"
  [ -s "$ENV_FILE" ] && [ -n "$(tail -n1 "$ENV_FILE")" ] && printf '\n' >> "$ENV_FILE"
}

# --- 1. CLI -----------------------------------------------------------------
if ! command -v az >/dev/null 2>&1; then
  say "installing Azure CLI"
  if command -v brew >/dev/null 2>&1; then brew install azure-cli; else
    curl -sL https://aka.ms/InstallAzureCLIDeb | bash || die "install az manually"; fi
fi
az extension add -n ml --yes >/dev/null 2>&1 || true

# --- 2. login ---------------------------------------------------------------
if ! az account show >/dev/null 2>&1; then say "az login (browser)"; az login || die "login failed"; fi
SUB="$(az account show --query id -o tsv)"
TENANT="$(az account show --query tenantId -o tsv)"
SCOPE="/subscriptions/$SUB/resourceGroups/$RG"
say "subscription $SUB (tenant $TENANT)"

# --- 3. region (respect the allowed-locations policy) -----------------------
ALLOWED="$(az policy assignment list -o json 2>/dev/null | python3 -c '
import sys, json
out = set()
for a in json.load(sys.stdin):
    for k, v in (a.get("parameters") or {}).items():
        if "location" in k.lower():
            val = (v or {}).get("value")
            if isinstance(val, list): out.update(val)
print(" ".join(sorted(out)))' 2>/dev/null)"
LOCATION="$PREF_LOCATION"
if [ -n "$ALLOWED" ]; then
  say "subscription restricts regions to: $ALLOWED"
  LOCATION=""
  for r in "$PREF_LOCATION" "${REGION_PREF[@]}"; do
    if [[ " $ALLOWED " == *" $r "* ]]; then LOCATION="$r"; break; fi
  done
  [ -z "$LOCATION" ] && LOCATION="$(echo "$ALLOWED" | tr ' ' '\n' | head -1)"
fi
say "using region: $LOCATION"

# --- 4. resource group + workspace ------------------------------------------
az group create -n "$RG" -l "$LOCATION" -o none && say "resource group $RG"
if az ml workspace show -n "$WS" -g "$RG" >/dev/null 2>&1; then
  say "workspace $WS already exists"
else
  say "creating workspace $WS (≈2-3 min)…"
  az ml workspace create -n "$WS" -g "$RG" -l "$LOCATION" -o none || die "workspace create failed"
fi

# --- 5. service principal (secret, else certificate) ------------------------
env_sep   # keep this run's AZURE_* keys as their own blank-line-separated block
# The SP gets `Contributor` — Azure's "everything EXCEPT IAM": it can create,
# modify and delete every resource in scope but CANNOT assign roles or manage
# access (Microsoft.Authorization/*/write is excluded). That mirrors AWS
# PowerUserAccess and GCP roles/editor — the interface boundary is enforced by
# the agent's exec gate, not RBAC, so scoping the SP's services only manufactures
# fake-negative runs. Scoped to the testbed resource group (`$SCOPE`), not the
# whole subscription, to bound blast radius. Contributor is control-plane only,
# so section 6 adds the data-plane role(s) it does not cover.
say "creating service principal $SP"
ERR="$(mktemp)"
SP_JSON="$(az ad sp create-for-rbac --name "$SP" --role Contributor --scopes "$SCOPE" -o json 2>"$ERR")"
if echo "$SP_JSON" | python3 -c 'import sys,json;json.load(sys.stdin)' 2>/dev/null && \
   [ -n "$(echo "$SP_JSON" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("password") or "")')" ]; then
  APPID="$(echo "$SP_JSON" | python3 -c 'import sys,json;print(json.load(sys.stdin)["appId"])')"
  PW="$(echo "$SP_JSON" | python3 -c 'import sys,json;print(json.load(sys.stdin)["password"])')"
  set_env AZURE_CLIENT_SECRET "$PW"
  say "using client SECRET auth"
else
  if grep -qi "Credential type not allowed" "$ERR"; then
    say "tenant blocks SP secrets — using a CERTIFICATE credential instead"
  else
    say "secret credential unavailable — trying a certificate"; fi
  mkdir -p "$TESTBED/.azure"; chmod 700 "$TESTBED/.azure"
  SP_JSON="$(az ad sp create-for-rbac --name "$SP" --create-cert --role Contributor --scopes "$SCOPE" -o json)" \
    || die "SP create (cert) failed"
  APPID="$(echo "$SP_JSON" | python3 -c 'import sys,json;print(json.load(sys.stdin)["appId"])')"
  PEM="$(echo "$SP_JSON" | python3 -c 'import sys,json;print(json.load(sys.stdin)["fileWithCertAndPrivateKey"])')"
  mv "$PEM" "$CERT_DST"; chmod 600 "$CERT_DST"
  set_env AZURE_CLIENT_CERTIFICATE_PATH "$CERT_DST"
  # ensure no stale secret lingers
  tmp="$(mktemp)"; grep -v "^AZURE_CLIENT_SECRET=" "$ENV_FILE" > "$tmp" 2>/dev/null || true; mv "$tmp" "$ENV_FILE"
  say "certificate saved to $CERT_DST (gitignored)"
fi
grep -qxF '.azure/' "$TESTBED/.gitignore" 2>/dev/null || echo '.azure/' >> "$TESTBED/.gitignore"

# --- 6. data-plane + monitoring roles (Contributor is control-plane only) ---
# Azure has no single "all data planes" role, so grant the data-plane roles the
# ML workloads actually touch: blob (datasets/artifacts/AML datastore) read+write
# and metrics. If a task floors on a different data plane (Key Vault secrets,
# Cosmos data, …), add that role here — same "avoid fake-negatives" spirit as
# Contributor; worth a live re-probe (cf. the GCP Vertex note).
for role in "Storage Blob Data Contributor" "Monitoring Reader"; do
  for attempt in 1 2 3 4 5; do
    if az role assignment create --assignee "$APPID" --role "$role" --scope "$SCOPE" -o none 2>/dev/null; then
      say "granted: $role"; break
    else sleep 10; fi   # SP propagation lag
  done
done

# --- 7. write .env ----------------------------------------------------------
set_env AZURE_TENANT_ID "$TENANT"
set_env AZURE_CLIENT_ID "$APPID"
set_env AZURE_SUBSCRIPTION_ID "$SUB"
set_env AZURE_RESOURCE_GROUP "$RG"
set_env AZUREML_WORKSPACE_NAME "$WS"
set_env AZURE_LOCATION "$LOCATION"

say "DONE. Validate with:  mlpab test configs/platforms/azure/sdk.yaml"
