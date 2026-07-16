#!/usr/bin/env bash
# One-time AWS bootstrap for the mlpab testbed. Like the gcp/azure bootstraps it
# provisions the AGENT identity automatically: you enter an ADMIN credential, it
# wires everything, and ONLY the minted agent key lands in testbed/.env.
# Idempotent — safe to re-run (re-running rotates the agent's access key).
#
# Unlike azure/gcp there is no CLI to install or `login` browser flow: AWS auth
# is an access-key pair. You provide an ADMIN pair at the prompt (used in-process
# for the IAM calls, NOT stored); botocore (added via uv if the base venv lacks
# it) then ensures, all automatically:
#
#   * mlpab-testbed-policy — the inline supplement (mlpab-policy.json) carrying
#     the one IAM action PowerUserAccess omits but the agent needs: iam:PassRole
#     on the execution role (jobs can't launch otherwise), plus the role
#     self-heal actions setup.py uses. Account id rewritten to the live account.
#   * The SageMaker execution role (mlpab-sagemaker-execution-role): trust +
#     AmazonSageMakerFullAccess + an inline ECR-pull policy (so jobs can pull
#     their container image on the very first run).
#   * The AGENT user (mlpab-agent, override via AGENT_USER): PowerUserAccess +
#     the supplement = "everything EXCEPT IAM". A fresh access key is minted for
#     it and written to .env. The interface gate — not IAM — is what confines the
#     agent to its interface, so a broad agent identity manufactures no escapes.
#
# Creating users/keys/policies/roles is IAM-admin work, deliberately separate
# from the PowerUser agent identity it mints — so run this with an ADMIN
# credential (simplest: an admin carrying the AWS-managed AdministratorAccess,
# or at least IAMFullAccess). The minted agent is PowerUser, never an admin.
#
#   bash configs/platforms/aws/bootstrap.sh        (or: make bootstrap-aws)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTBED="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ENV_FILE="$TESTBED/.env"
POLICY_JSON="$SCRIPT_DIR/mlpab-policy.json"
PY="$TESTBED/.venv/bin/python"   # base venv python (botocore added via uv if absent)
AGENT_USER="${AGENT_USER:-mlpab-agent}"   # the PowerUser identity the agent runs as

say() { printf '\n>> %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

[ -x "$PY" ] || die "base venv python not found at $PY — run \`make install\` first."
[ -f "$POLICY_JSON" ] || die "policy file missing: $POLICY_JSON"

set_env() {  # set_env KEY VALUE — replace existing line or append; value-safe
  local key="$1"; shift; local val="$*" tmp
  # Strip CR/LF: a pasted secret with a stray newline must NOT split across two
  # .env lines (that orphans the tail as a bare, keyless line and breaks the
  # value). Keys/paths/secrets are always single-line.
  val="${val//$'\r'/}"; val="${val//$'\n'/}"
  tmp="$(mktemp)"; touch "$ENV_FILE"
  # Drop the old line for this key AND squeeze runs of blank lines to one, so
  # re-runs don't accumulate gaps.
  grep -v "^${key}=" "$ENV_FILE" 2>/dev/null | cat -s > "$tmp" || true
  printf '%s=%s\n' "$key" "$val" >> "$tmp"
  mv "$tmp" "$ENV_FILE"
}

env_sep() {  # one blank-line separator before a new bootstrap's block (idempotent)
  touch "$ENV_FILE"
  [ -s "$ENV_FILE" ] && [ -n "$(tail -c1 "$ENV_FILE")" ] && printf '\n' >> "$ENV_FILE"
  [ -s "$ENV_FILE" ] && [ -n "$(tail -n1 "$ENV_FILE")" ] && printf '\n' >> "$ENV_FILE"
}

env_get() {  # current value: existing process env wins, else the .env line
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
  # NOTE: the trailing newline after a silent read MUST go to stderr (>&2): this
  # function runs inside $(...), so an `echo` to stdout would be captured into
  # the value, prefixing secrets with a newline (which split them across .env
  # lines / broke their signature). The read prompt itself already goes to stderr.
  if [ -n "$secret" ]; then read -rsp "  $msg: " input; echo >&2; else read -rp "  $msg: " input; fi
  printf '%s' "${input:-$cur}"
}

# --- 1. ADMIN credentials (used ONCE for setup, NOT written to .env) ---------
# Like the gcp/azure bootstraps, the AGENT identity is provisioned automatically
# (section 2): you enter an ADMIN credential here, it mints a dedicated PowerUser
# agent user + access key, and only THAT agent key lands in .env. The admin
# creds are passed to the IAM step in-process and never stored. The admin needs
# IAM-admin rights (simplest: AWS-managed AdministratorAccess / IAMFullAccess).
say "AWS ADMIN credentials (used once for setup; NOT stored in $ENV_FILE)"
ADMIN_ACCESS_KEY_ID="$(prompt ADMIN_ACCESS_KEY_ID 'ADMIN AWS_ACCESS_KEY_ID')"
ADMIN_SECRET_ACCESS_KEY="$(prompt ADMIN_SECRET_ACCESS_KEY 'ADMIN AWS_SECRET_ACCESS_KEY' secret)"
AWS_REGION="$(prompt AWS_REGION 'AWS_REGION (e.g. us-east-1)')"
AWS_REGION="${AWS_REGION:-us-east-1}"
[ -n "$ADMIN_ACCESS_KEY_ID" ] && [ -n "$ADMIN_SECRET_ACCESS_KEY" ] || die "admin access key id + secret are required."

# --- 2. provision everything with the admin creds (via botocore) ------------
# botocore is NOT in the base venv (and must not be — the AWS interface pins its
# own botocore per run; a base-venv copy risks per-run install conflicts). This
# one-shot step needs botocore, so run it in an EPHEMERAL uv env when the base
# venv lacks it — no base-venv pollution, no new declared dependency.
if "$PY" -c 'import botocore' 2>/dev/null; then
  IAM_PY=("$PY")
elif command -v uv >/dev/null 2>&1; then
  IAM_PY=(uv run --no-project --with botocore python)
  say "base venv has no botocore — running the setup step via an ephemeral uv env"
else
  die "the setup step needs botocore: install uv (\`brew install uv\`) or \`$PY -m pip install botocore\`."
fi
say "provisioning supplement policy + execution role + PowerUser agent user"
AGENT_CRED_OUT="$(mktemp)"   # python writes the minted <id>\n<secret> here
trap 'rm -f "$AGENT_CRED_OUT"' EXIT   # never leak the cred file, even on `die`
# Admin creds drive the IAM calls (botocore reads AWS_*); they are NOT stored.
AWS_ACCESS_KEY_ID="$ADMIN_ACCESS_KEY_ID" \
AWS_SECRET_ACCESS_KEY="$ADMIN_SECRET_ACCESS_KEY" \
AWS_REGION="$AWS_REGION" \
MLPAB_POLICY_JSON="$POLICY_JSON" \
MLPAB_AGENT_USER="$AGENT_USER" \
MLPAB_AGENT_CRED_OUT="$AGENT_CRED_OUT" \
"${IAM_PY[@]}" - <<'PYEOF' || die "setup failed (run with admin creds — AdministratorAccess / IAMFullAccess)."
import json, os
import botocore.session

POLICY_NAME = "mlpab-testbed-policy"
ROLE_NAME = "mlpab-sagemaker-execution-role"
AGENT_USER = os.environ["MLPAB_AGENT_USER"]
EXEC_POLICY = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
# The agent gets unrestricted access EXCEPT IAM/Organizations/Account: the
# interface boundary is enforced by the agent gate, not IAM, so scoping the
# agent's services only manufactures fake-negative runs. PowerUserAccess covers
# everything; the inline supplement adds back only the IAM action it omits but
# the agent needs — iam:PassRole on the execution role (+ role self-heal for
# setup.py). IAM management stays denied to the agent.
POWERUSER_POLICY = "arn:aws:iam::aws:policy/PowerUserAccess"
ECR_PULL_POLICY_NAME = "mlpab-ecr-pull"
# SageMaker jobs pull their container image from AWS-owned ECR; the execution
# role must be allowed to. Set it explicitly here so the role works on the very
# first run (setup.py also re-ensures it per run).
ECR_PULL_POLICY = {"Version": "2012-10-17", "Statement": [
    {"Effect": "Allow", "Action": "ecr:GetAuthorizationToken", "Resource": "*"},
    {"Effect": "Allow", "Action": ["ecr:BatchCheckLayerAvailability",
     "ecr:GetDownloadUrlForLayer", "ecr:BatchGetImage"],
     "Resource": "arn:aws:ecr:*:*:repository/*"}]}
TRUST = {"Version": "2012-10-17", "Statement": [{
    "Effect": "Allow", "Principal": {"Service": "sagemaker.amazonaws.com"},
    "Action": "sts:AssumeRole"}]}

session = botocore.session.get_session()
region = os.environ.get("AWS_REGION")
account = session.create_client("sts", region_name=region).get_caller_identity()["Account"]
iam = session.create_client("iam", region_name=region)
print(f"   admin account: {account}")

# 1) supplement policy (PassRole + execution-role self-heal for setup.py).
doc = open(os.environ["MLPAB_POLICY_JSON"]).read()
import re
doc = re.sub(r'(arn:aws:[^"]*?::?)\d{12}(:)', lambda m: f"{m.group(1)}{account}{m.group(2)}", doc)
json.loads(doc)  # validate
# Guard: the regex only rewrites ARNs with a trailing-colon account field (IAM,
# ECR, …); an S3-style ARN (account inside the bucket name) would slip through.
# Fail loudly rather than upload a policy pinned to the wrong (placeholder) account.
if account != "710271938540" and "710271938540" in doc:
    raise SystemExit("supplement policy still references the placeholder account "
                     "710271938540 after rewrite — fix the ARN form in mlpab-policy.json")
policy_arn = f"arn:aws:iam::{account}:policy/{POLICY_NAME}"
try:
    iam.create_policy(PolicyName=POLICY_NAME, PolicyDocument=doc,
                      Description="mlpab testbed: IAM actions PowerUserAccess omits (PassRole + role self-heal)")
    print(f"   created policy {policy_arn}")
except iam.exceptions.EntityAlreadyExistsException:
    versions = iam.list_policy_versions(PolicyArn=policy_arn)["Versions"]
    nondefault = sorted((v for v in versions if not v["IsDefaultVersion"]), key=lambda v: v["CreateDate"])
    while len(versions) >= 5 and nondefault:   # AWS caps a policy at 5 versions
        iam.delete_policy_version(PolicyArn=policy_arn, VersionId=nondefault.pop(0)["VersionId"])
        versions.pop()
    iam.create_policy_version(PolicyArn=policy_arn, PolicyDocument=doc, SetAsDefault=True)
    print(f"   updated policy {policy_arn} (new default version)")

# 2) execution role: trust SageMaker + SageMakerFullAccess + ECR pull.
try:
    iam.get_role(RoleName=ROLE_NAME)
    print(f"   execution role {ROLE_NAME} exists")
except iam.exceptions.NoSuchEntityException:
    iam.create_role(RoleName=ROLE_NAME, AssumeRolePolicyDocument=json.dumps(TRUST),
                    Description="mlpab testbed: execution role assumed by SageMaker jobs")
    print(f"   created execution role {ROLE_NAME}")
iam.attach_role_policy(RoleName=ROLE_NAME, PolicyArn=EXEC_POLICY)
iam.put_role_policy(RoleName=ROLE_NAME, PolicyName=ECR_PULL_POLICY_NAME,
                    PolicyDocument=json.dumps(ECR_PULL_POLICY))
print(f"   ensured AmazonSageMakerFullAccess + ECR pull on {ROLE_NAME}")

# 3) agent user = PowerUser + supplement (everything except IAM).
try:
    iam.get_user(UserName=AGENT_USER)
    print(f"   agent user {AGENT_USER} exists")
except iam.exceptions.NoSuchEntityException:
    iam.create_user(UserName=AGENT_USER, Tags=[{"Key": "mlpab", "Value": "agent"}])
    print(f"   created agent user {AGENT_USER}")
iam.attach_user_policy(UserName=AGENT_USER, PolicyArn=POWERUSER_POLICY)
iam.attach_user_policy(UserName=AGENT_USER, PolicyArn=policy_arn)
print(f"   attached PowerUserAccess + {POLICY_NAME} to {AGENT_USER}")

# 4) mint a fresh access key (rotate if already at AWS's 2-key cap).
keys = iam.list_access_keys(UserName=AGENT_USER)["AccessKeyMetadata"]
if len(keys) >= 2:
    oldest = sorted(keys, key=lambda k: k["CreateDate"])[0]["AccessKeyId"]
    iam.delete_access_key(UserName=AGENT_USER, AccessKeyId=oldest)
    print(f"   rotated out oldest access key {oldest} (2-key cap)")
new = iam.create_access_key(UserName=AGENT_USER)["AccessKey"]
with open(os.environ["MLPAB_AGENT_CRED_OUT"], "w") as f:
    f.write(new["AccessKeyId"] + "\n" + new["SecretAccessKey"] + "\n")
print(f"   minted access key {new['AccessKeyId']} for {AGENT_USER}")
PYEOF

# Write ONLY the minted agent key to .env (admin creds are never stored).
AGENT_ID="$(sed -n 1p "$AGENT_CRED_OUT")"
AGENT_SECRET="$(sed -n 2p "$AGENT_CRED_OUT")"
rm -f "$AGENT_CRED_OUT"
[ -n "$AGENT_ID" ] && [ -n "$AGENT_SECRET" ] || die "agent access key was not minted."
env_sep   # this run's AWS_* keys as their own blank-line-separated block
set_env AWS_ACCESS_KEY_ID "$AGENT_ID"
set_env AWS_SECRET_ACCESS_KEY "$AGENT_SECRET"
set_env AWS_REGION "$AWS_REGION"
say "wrote agent AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION (user $AGENT_USER, PowerUser)"

say "DONE. Validate with:  mlpab check configs/treatments/aws/aws-claude.yaml"
say "      (or a single interface:  mlpab test configs/platforms/aws/cli.yaml)"
say "      NOTE: a freshly minted access key can take a few seconds to activate."
