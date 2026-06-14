#!/usr/bin/env bash
# One-time AWS bootstrap for the mlpab testbed: collect the AWS_* credentials,
# create + attach the mlpab IAM policy (configs/platforms/aws/mlpab-policy.json)
# to the calling IAM user, and pre-provision the SageMaker execution role.
# Idempotent — safe to re-run.
#
# Unlike azure/gcp there is no CLI to install or `login` browser flow: AWS auth
# is the AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION key triple. This
# script collects those into testbed/.env, then uses botocore (already in the
# base .venv) to ensure the IAM plumbing the testbed needs:
#
#   * The managed policy from mlpab-policy.json — grants the testbed user the
#     rights setup.py's role bootstrap and the live auth_command depend on
#     (iam:GetRole/CreateRole/AttachRolePolicy + PassRole on the execution role,
#     servicequotas discovery, s3vectors, sagemaker log access). The policy's
#     account id is rewritten to the caller's live account before upload.
#   * The execution role (mlpab-sagemaker-execution-role) SageMaker assumes —
#     setup.py also ensures this per run, but creating it here makes the very
#     first run ready immediately.
#
# Attaching a policy to yourself needs iam:CreatePolicy + iam:AttachUserPolicy,
# so run this with credentials that carry IAM admin rights (a one-time step).
#
#   bash configs/platforms/aws/bootstrap.sh        (or: make bootstrap-aws)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTBED="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ENV_FILE="$TESTBED/.env"
POLICY_JSON="$SCRIPT_DIR/mlpab-policy.json"
PY="$TESTBED/.venv/bin/python"   # botocore lives in the base venv

say() { printf '\n>> %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

[ -x "$PY" ] || die "base venv python not found at $PY — run \`make install\` first."
[ -f "$POLICY_JSON" ] || die "policy file missing: $POLICY_JSON"

set_env() {  # set_env KEY VALUE — replace existing line or append; value-safe
  local key="$1"; shift; local val="$*" tmp
  tmp="$(mktemp)"; touch "$ENV_FILE"
  grep -v "^${key}=" "$ENV_FILE" > "$tmp" 2>/dev/null || true
  printf '%s=%s\n' "$key" "$val" >> "$tmp"
  mv "$tmp" "$ENV_FILE"
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
  if [ -n "$secret" ]; then read -rsp "  $msg: " input; echo; else read -rp "  $msg: " input; fi
  printf '%s' "${input:-$cur}"
}

# --- 1. credentials ---------------------------------------------------------
say "AWS credentials (written to $ENV_FILE)"
AWS_ACCESS_KEY_ID="$(prompt AWS_ACCESS_KEY_ID 'AWS_ACCESS_KEY_ID')"
AWS_SECRET_ACCESS_KEY="$(prompt AWS_SECRET_ACCESS_KEY 'AWS_SECRET_ACCESS_KEY' secret)"
AWS_REGION="$(prompt AWS_REGION 'AWS_REGION (e.g. us-east-1)')"
AWS_REGION="${AWS_REGION:-us-east-1}"
[ -n "$AWS_ACCESS_KEY_ID" ] && [ -n "$AWS_SECRET_ACCESS_KEY" ] || die "access key id + secret are required."
set_env AWS_ACCESS_KEY_ID "$AWS_ACCESS_KEY_ID"
set_env AWS_SECRET_ACCESS_KEY "$AWS_SECRET_ACCESS_KEY"
set_env AWS_REGION "$AWS_REGION"
say "wrote AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION"

# --- 2. IAM: policy + execution role (via botocore) -------------------------
say "ensuring IAM policy + execution role"
AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
AWS_REGION="$AWS_REGION" \
MLPAB_POLICY_JSON="$POLICY_JSON" \
"$PY" - <<'PYEOF' || die "IAM bootstrap failed (need IAM admin rights on these credentials)."
import json, os, sys
import botocore.session

POLICY_NAME = "mlpab-testbed-policy"
ROLE_NAME = "mlpab-sagemaker-execution-role"
EXEC_POLICY = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
TRUST = {"Version": "2012-10-17", "Statement": [{
    "Effect": "Allow", "Principal": {"Service": "sagemaker.amazonaws.com"},
    "Action": "sts:AssumeRole"}]}

session = botocore.session.get_session()
region = os.environ.get("AWS_REGION")
sts = session.create_client("sts", region_name=region)
iam = session.create_client("iam", region_name=region)

ident = sts.get_caller_identity()
account, arn = ident["Account"], ident["Arn"]
print(f"   caller: {arn}")

doc = open(os.environ["MLPAB_POLICY_JSON"]).read()
# Rewrite the committed account id to this caller's live account so the policy's
# resource ARNs are correct in any account.
import re
doc = re.sub(r'(arn:aws:[^"]*?::?)\d{12}(:)', lambda m: f"{m.group(1)}{account}{m.group(2)}", doc)
json.loads(doc)  # validate

policy_arn = f"arn:aws:iam::{account}:policy/{POLICY_NAME}"
try:
    iam.create_policy(PolicyName=POLICY_NAME, PolicyDocument=doc,
                      Description="mlpab testbed: rights for SageMaker role bootstrap, quotas, s3vectors, logs")
    print(f"   created policy {policy_arn}")
except iam.exceptions.EntityAlreadyExistsException:
    versions = iam.list_policy_versions(PolicyArn=policy_arn)["Versions"]
    nondefault = sorted((v for v in versions if not v["IsDefaultVersion"]),
                        key=lambda v: v["CreateDate"])
    while len(versions) >= 5 and nondefault:   # AWS caps a policy at 5 versions
        old = nondefault.pop(0)
        iam.delete_policy_version(PolicyArn=policy_arn, VersionId=old["VersionId"])
        versions = [v for v in versions if v["VersionId"] != old["VersionId"]]
    iam.create_policy_version(PolicyArn=policy_arn, PolicyDocument=doc, SetAsDefault=True)
    print(f"   updated policy {policy_arn} (new default version)")

if ":user/" in arn:
    user = arn.split(":user/", 1)[1]
    iam.attach_user_policy(UserName=user, PolicyArn=policy_arn)
    print(f"   attached {POLICY_NAME} to user {user}")
else:
    print(f"   NOTE: caller is not an IAM user — attach {policy_arn} to your principal manually")

# Execution role (idempotent). setup.py also ensures this per run.
try:
    r = iam.get_role(RoleName=ROLE_NAME)
    print(f"   execution role exists: {r['Role']['Arn']}")
except iam.exceptions.NoSuchEntityException:
    r = iam.create_role(RoleName=ROLE_NAME, AssumeRolePolicyDocument=json.dumps(TRUST),
                        Description="mlpab testbed: execution role assumed by SageMaker jobs")
    iam.attach_role_policy(RoleName=ROLE_NAME, PolicyArn=EXEC_POLICY)
    print(f"   created execution role {r['Role']['Arn']}")
PYEOF

say "DONE. Validate with:  mlpab check configs/treatments/aws/aws-claude.yaml"
say "      (or a single interface:  mlpab test configs/platforms/aws/cli.yaml)"
