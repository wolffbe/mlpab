#!/usr/bin/env bash
# One-time GCP bootstrap for the mlpab testbed: install gcloud, log in, enable
# APIs, create the service account + roles + BigQuery dataset, and write the
# GCP_* keys into testbed/.env. Idempotent — safe to re-run.
#
# Encodes the real-world gotcha this org hit:
#   * The org may FORBID service-account keys
#     (constraints/iam.disableServiceAccountKeyCreation) — we fall back to
#     keyless Application Default Credentials (ADC), impersonating the SA
#     (GCP-recommended), and point GOOGLE_APPLICATION_CREDENTIALS at the ADC.
#
#   PROJECT=my-project bash configs/platforms/gcp/bootstrap.sh
# Overridable via env: PROJECT (else gcloud's default), SA, DATASET, LOCATION.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTBED="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ENV_FILE="$TESTBED/.env"
KEY_DST="$TESTBED/.gcp/sa-key.json"
ADC_DST="$TESTBED/.gcp/adc.json"

SA="${SA:-mlpab-sa}"
DATASET="${DATASET:-mlpab}"
LOCATION="${LOCATION:-***REDACTED***}"   # EU by default; Vertex + Vector Search supported
# The interface boundary is enforced by the agent HOOK (the on-interface
# `gcloud`/SDK surface), NOT by IAM — so scoping the service account's roles
# adds no experimental value and only manufactures fake-negative runs when a
# task needs a resource outside the curated set (the GCP analog of the AWS
# execution-role ECR gap seen live 2026-06-14). So grant the basic `roles/editor`
# (GCP's "unrestricted except IAM": modify essentially every resource, but
# cannot set IAM policy, manage roles, or touch billing/org). The keyless-ADC
# fallback below still adds only the impersonation grant the user needs.
ROLES=(roles/editor)

say() { printf '\n>> %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }
set_env() { local key="$1"; shift; local val="$*" tmp; tmp="$(mktemp)"; touch "$ENV_FILE"
  val="${val//$'\r'/}"; val="${val//$'\n'/}"   # never split a value across lines
  grep -v "^${key}=" "$ENV_FILE" 2>/dev/null | cat -s > "$tmp" || true
  printf '%s=%s\n' "$key" "$val" >> "$tmp"; mv "$tmp" "$ENV_FILE"; }
env_sep() {  # one blank-line separator before this bootstrap's block (idempotent)
  touch "$ENV_FILE"
  [ -s "$ENV_FILE" ] && [ -n "$(tail -c1 "$ENV_FILE")" ] && printf '\n' >> "$ENV_FILE"
  [ -s "$ENV_FILE" ] && [ -n "$(tail -n1 "$ENV_FILE")" ] && printf '\n' >> "$ENV_FILE"; }

# --- 1. CLI -----------------------------------------------------------------
if ! command -v gcloud >/dev/null 2>&1; then
  say "installing Google Cloud SDK"
  if command -v brew >/dev/null 2>&1; then brew install --cask google-cloud-sdk; else
    curl -sSL https://sdk.cloud.google.com | bash || die "install gcloud manually"
    exec -l "$SHELL"; fi
fi

# --- 2. login + project -----------------------------------------------------
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
  say "gcloud auth login (browser)"; gcloud auth login || die "login failed"; fi
PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
[ -n "$PROJECT" ] && [ "$PROJECT" != "(unset)" ] || die "set PROJECT=... (no default project)"
gcloud config set project "$PROJECT" >/dev/null 2>&1
SA_EMAIL="$SA@$PROJECT.iam.gserviceaccount.com"
say "project $PROJECT"

# --- 3. APIs ----------------------------------------------------------------
say "enabling APIs"
gcloud services enable aiplatform.googleapis.com bigquery.googleapis.com \
  storage.googleapis.com monitoring.googleapis.com --project "$PROJECT" || die "enable APIs failed"

# --- 4. service account + roles ---------------------------------------------
gcloud iam service-accounts describe "$SA_EMAIL" --project "$PROJECT" >/dev/null 2>&1 \
  || { say "creating service account $SA"; gcloud iam service-accounts create "$SA" \
         --display-name "mlpab testbed agent" --project "$PROJECT"; }
for role in "${ROLES[@]}"; do
  gcloud projects add-iam-policy-binding "$PROJECT" --member "serviceAccount:$SA_EMAIL" \
    --role "$role" --condition=None -q >/dev/null 2>&1 && say "granted: $role"
done

# --- 5. BigQuery dataset (EU) -----------------------------------------------
if bq --project_id="$PROJECT" show -d "$DATASET" >/dev/null 2>&1; then
  say "dataset $DATASET already exists"
else
  say "creating BigQuery dataset $DATASET in $LOCATION"
  bq --location="$LOCATION" --project_id="$PROJECT" mk -d "$DATASET" || die "dataset create failed"
fi

# --- 6. credentials: SA key, else keyless ADC (org may block keys) ----------
env_sep   # keep this run's GCP_* / GOOGLE_* keys as their own separated block
mkdir -p "$TESTBED/.gcp"; chmod 700 "$TESTBED/.gcp"
grep -qxF '.gcp/' "$TESTBED/.gitignore" 2>/dev/null || echo '.gcp/' >> "$TESTBED/.gitignore"
ERR="$(mktemp)"
if gcloud iam service-accounts keys create "$KEY_DST" --iam-account "$SA_EMAIL" \
     --project "$PROJECT" 2>"$ERR" && [ -s "$KEY_DST" ]; then
  chmod 600 "$KEY_DST"; set_env GOOGLE_APPLICATION_CREDENTIALS "$KEY_DST"
  say "SA key saved to $KEY_DST (gitignored)"
else
  rm -f "$KEY_DST"
  if grep -qi "disableServiceAccountKeyCreation\|not allowed" "$ERR"; then
    say "org blocks SA keys — using keyless ADC with SA impersonation"
  else say "SA key unavailable — using keyless ADC"; fi
  USER_ACCT="$(gcloud config get-value account 2>/dev/null)"
  gcloud iam service-accounts add-iam-policy-binding "$SA_EMAIL" \
    --member "user:$USER_ACCT" --role roles/iam.serviceAccountTokenCreator --project "$PROJECT" -q >/dev/null 2>&1 || true
  say "running ADC login (browser) impersonating $SA_EMAIL"
  gcloud auth application-default login --impersonate-service-account="$SA_EMAIL" \
    || gcloud auth application-default login || die "ADC login failed"
  ADC_SRC="$HOME/.config/gcloud/application_default_credentials.json"
  [ -f "$ADC_SRC" ] || die "ADC file not found after login"
  cp "$ADC_SRC" "$ADC_DST"; chmod 600 "$ADC_DST"
  set_env GOOGLE_APPLICATION_CREDENTIALS "$ADC_DST"
  say "ADC saved to $ADC_DST (gitignored)"
fi

# --- 7. write .env ----------------------------------------------------------
set_env GCP_PROJECT "$PROJECT"
set_env GCP_LOCATION "$LOCATION"
set_env GCP_BQ_DATASET "$DATASET"

say "DONE. Validate with:  mlpab test configs/platforms/gcp/sdk.yaml"
