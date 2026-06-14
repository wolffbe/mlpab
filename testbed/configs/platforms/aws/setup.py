"""SageMaker platform setup — ensure the platform plumbing the agent cannot create.

Run automatically by `mlpab run` (the interface `serve:` step) at the START of
every challenge, right AFTER `teardown.py` has swept the account — same contract
as the hopsworks setup.

Unlike Hopsworks there is NO project container to pre-create: AWS credentials
scope to the (account, region) directly and every interface authenticates
without any pre-existing resource. What IS worth guaranteeing is the plumbing
SageMaker jobs need but the engineer is locked out of creating (S3 and IAM are
both OFF-INTERFACE — only `aws sagemaker` / `import sagemaker` / the MCP tools
are sanctioned):

  1. The default SageMaker bucket `sagemaker-<region>-<account>`. Jobs write
     their artifacts to S3 paths the agent must supply, and the bucket those
     paths live in has to already exist. Pre-creating the well-known default
     (the same bucket the sagemaker SDK auto-creates on first use) gives every
     interface a working artifact location without the agent touching S3.
     `teardown.py` empties this bucket at run start+end but deliberately keeps
     the bucket itself.
  2. The IAM EXECUTION ROLE. Every job-creating call (training, processing,
     endpoints, …) requires a RoleArn that SageMaker assumes server-side; the
     agent can neither create nor discover one (IAM/STS are off-interface).
     This script ensures the role exists — creating it with the SageMaker
     trust policy + AmazonSageMakerFullAccess when the testbed credentials
     carry IAM rights (that managed policy grants S3 access to buckets with
     "sagemaker" in the name, which covers bucket #1) — and EXPORTS its ARN to
     the engineer's env as SAGEMAKER_ROLE_ARN via $MLPAB_PLATFORM_ENV (the
     KEY=VALUE handoff file mlpab merges into the run env; the base prompt
     names the variable). No .env entry is needed; setting SAGEMAKER_ROLE_ARN
     in .env anyway overrides the export and picks the role name to ensure.
     `teardown.py` leaves the role alone — plumbing, not agent state.

Neither is part of the measured FTI lifecycle. Uses botocore directly (the one
AWS lib present in all three run venvs).

Best-effort by design: invoked via the interface `serve:` step, whose runner
(`_run_aux`) ignores failures and discards output. Nothing here may raise out of
`main()` — a setup hiccup must never fail an engineer run.
"""

from __future__ import annotations

import json
import os

# Role name used when SAGEMAKER_ROLE_ARN is unset (the ARN it implies is
# arn:aws:iam::<account>:role/mlpab-sagemaker-execution-role — set that in
# .env so the engineer's env and prompt actually carry it).
DEFAULT_ROLE_NAME = "mlpab-sagemaker-execution-role"

TRUST_POLICY = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"Service": "sagemaker.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }
    ],
}
EXECUTION_POLICY_ARN = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"


def _ensure_execution_role(session, region: str) -> str | None:
    """Make sure the execution role SAGEMAKER_ROLE_ARN names exists (create if
    the credentials allow IAM) and return its CONFIRMED ARN, else None."""
    arn = os.environ.get("SAGEMAKER_ROLE_ARN") or ""
    role_name = arn.rsplit("/", 1)[-1] if arn else DEFAULT_ROLE_NAME
    try:
        iam = session.create_client("iam", region_name=region)
    except Exception as e:
        print(f"[sagemaker setup] iam client unavailable: {e}")
        return None

    try:
        resp = iam.get_role(RoleName=role_name)
        print(f"[sagemaker setup] execution role {role_name!r} already exists")
        return (resp.get("Role") or {}).get("Arn")
    except Exception:
        pass  # missing (or no iam:GetRole) → try to create it

    try:
        resp = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(TRUST_POLICY),
            Description="mlpab testbed: execution role assumed by SageMaker jobs",
        )
        iam.attach_role_policy(RoleName=role_name, PolicyArn=EXECUTION_POLICY_ARN)
        created_arn = (resp.get("Role") or {}).get("Arn")
        print(f"[sagemaker setup] created execution role {role_name!r} ({created_arn})")
        return created_arn
    except Exception as e:
        # Credentials without IAM rights → the role must be pre-provisioned by
        # hand; the engineer run can still proceed (and fail loudly on RoleArn).
        print(f"[sagemaker setup] create of execution role {role_name!r} skipped: {e}")
        return None


def _export_for_engineer(key: str, value: str) -> None:
    """Hand a value to the engineer's env: mlpab merges KEY=VALUE lines appended
    to $MLPAB_PLATFORM_ENV into the run env (declared .env keys win)."""
    path = os.environ.get("MLPAB_PLATFORM_ENV")
    if not path:
        return
    try:
        with open(path, "a") as f:
            f.write(f"{key}={value}\n")
        print(f"[sagemaker setup] exported {key} to the engineer env")
    except Exception as e:
        print(f"[sagemaker setup] export of {key} skipped: {e}")


def main() -> None:
    try:
        import botocore.session
    except Exception as e:  # no AWS lib in this venv → nothing to do
        print(f"[sagemaker setup] botocore unavailable: {e}")
        return

    session = botocore.session.get_session()
    region = os.environ.get("AWS_REGION") or session.get_config_variable("region")
    try:
        sts = session.create_client("sts", region_name=region)
        account = sts.get_caller_identity()["Account"]
        s3 = session.create_client("s3", region_name=region)
    except Exception as e:
        # No reachable AWS / bad creds → preflight login will report it.
        print(f"[sagemaker setup] aws client unavailable: {e}")
        return

    bucket = f"sagemaker-{region}-{account}"
    try:
        s3.head_bucket(Bucket=bucket)
        print(f"[sagemaker setup] default bucket {bucket!r} already exists")
    except Exception:
        # Missing (or not visible) → try to create it.
        try:
            params = {"Bucket": bucket}
            if region != "us-east-1":  # us-east-1 rejects a LocationConstraint
                params["CreateBucketConfiguration"] = {"LocationConstraint": region}
            s3.create_bucket(**params)
            print(f"[sagemaker setup] created default bucket {bucket!r}")
        except Exception as e:
            print(f"[sagemaker setup] create of bucket {bucket!r} skipped: {e}")
    # The engineer stages data/artifacts here via `aws s3` (and the prompt pins
    # it to this bucket — teardown.py only sweeps THIS bucket's contents).
    _export_for_engineer("SAGEMAKER_DEFAULT_BUCKET", bucket)

    role_arn = _ensure_execution_role(session, region)
    if role_arn:
        _export_for_engineer("SAGEMAKER_ROLE_ARN", role_arn)


def verify() -> int:
    """Twofold setup check (`setup.py verify`): (1) AWS CONNECTS (sts), then (2)
    the default SageMaker bucket setup guarantees is PRESENT. Connection is the
    hard gate (non-zero fails the run); the bucket is best-effort. Read-only."""
    try:
        import botocore.session
    except Exception as e:
        print(f"[sagemaker verify-setup] botocore unavailable: {e}")
        return 1
    session = botocore.session.get_session()
    region = os.environ.get("AWS_REGION") or session.get_config_variable("region")
    try:
        account = session.create_client("sts", region_name=region).get_caller_identity()["Account"]
    except Exception as e:
        print(f"[sagemaker verify-setup] NO CONNECTION: {e}")
        return 1
    bucket = f"sagemaker-{region}-{account}"
    try:
        session.create_client("s3", region_name=region).head_bucket(Bucket=bucket)
        print(f"[sagemaker verify-setup] OK: connected (acct {account}); bucket {bucket!r} present")
    except Exception as e:
        print(
            f"[sagemaker verify-setup] OK: connected; bucket {bucket!r} absent "
            f"(best-effort, not failing): {e}"
        )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(verify() if sys.argv[1:2] == ["verify"] else (main() or 0))
