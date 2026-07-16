# Databricks notebook source
# Upload finetuned_model.npz content to the EXISTING pending version 1, then
# finalize + attach metric tags. Tries several SSE strategies for the managed
# S3 bucket's resource policy.
import json, boto3
from botocore.exceptions import ClientError
from databricks.sdk import WorkspaceClient

STAGE = "/Volumes/workspace/mlpab5b087a/ftstage"
FULL = "workspace.mlpab5b087a.ftmodel698f06"
VERSION = 1
w = WorkspaceClient(); ac = w.api_client
R = {"attempts": []}

metrics = json.load(open(f"{STAGE}/metrics.json")); R["metrics"] = metrics
loc = w.model_versions.get(FULL, VERSION).storage_location
R["loc"] = loc

creds = ac.do("POST", "/api/2.0/mlflow/unity-catalog/model-versions/generate-temporary-credentials",
              body={"name": FULL, "version": VERSION, "operation": "MODEL_VERSION_OPERATION_READ_WRITE"})
aws = creds["credentials"]["aws_temp_credentials"]
s3 = boto3.client("s3",
    aws_access_key_id=aws["access_key_id"],
    aws_secret_access_key=aws["secret_access_key"],
    aws_session_token=aws["session_token"],
    region_name="us-west-2")
bucket, _, key_prefix = loc[len("s3://"):].partition("/")
key = f"{key_prefix.rstrip('/')}/finetuned_model.npz"
body = open(f"{STAGE}/finetuned_model.npz", "rb").read()

strategies = [
    {"ServerSideEncryption": "aws:kms"},
    {"ServerSideEncryption": "AES256"},
    {"ServerSideEncryption": "aws:kms", "BucketKeyEnabled": True},
    {},
]
uploaded = False
for extra in strategies:
    try:
        s3.put_object(Bucket=bucket, Key=key, Body=body, **extra)
        R["attempts"].append([str(extra), "ok"]); uploaded = True; break
    except ClientError as e:
        R["attempts"].append([str(extra), str(e.response.get("Error", {}).get("Code"))])
R["uploaded"] = uploaded

if uploaded:
    fin = ac.do("POST", "/api/2.0/mlflow/unity-catalog/model-versions/finalize",
                body={"name": FULL, "version": VERSION})
    R["finalized_status"] = fin.get("model_version", {}).get("status")
    for k, v in metrics.items():
        try:
            ac.do("POST", "/api/2.0/mlflow/unity-catalog/model-versions/set-tag",
                  body={"name": FULL, "version": VERSION, "key": k, "value": str(v)})
            R["attempts"].append([f"tag:{k}", "ok"])
        except Exception as e:
            R["attempts"].append([f"tag:{k}", repr(e)[:150]])
    try:
        w.registered_models.set_alias(FULL, "prod", VERSION); R["alias"] = "ok"
    except Exception as e:
        R["alias"] = repr(e)[:150]

dbutils.notebook.exit(json.dumps(R))
