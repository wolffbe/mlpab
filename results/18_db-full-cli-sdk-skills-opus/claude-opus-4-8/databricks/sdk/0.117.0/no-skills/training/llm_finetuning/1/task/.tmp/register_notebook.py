# Databricks notebook source
# Register the fine-tuned model into the Unity Catalog model registry as
# workspace.mlpab5b087a.ftmodel698f06 version 1, with finetuned_model.npz as the
# model content and metrics.json attached as tags. Serverless (Databricks
# compute); uses the MLflow-UC REST API + boto3 (mlflow not installed here).
import json, boto3
from databricks.sdk import WorkspaceClient

STAGE = "/Volumes/workspace/mlpab5b087a/ftstage"
CAT, SCH, NAME = "workspace", "mlpab5b087a", "ftmodel698f06"
FULL = f"{CAT}.{SCH}.{NAME}"
CREDS_EP = "/api/2.0/mlflow/unity-catalog/model-versions/generate-temporary-credentials"

w = WorkspaceClient()
ac = w.api_client
R = {"steps": []}

def step(name, fn):
    try:
        v = fn(); R["steps"].append([name, "ok"]); return v
    except Exception as e:
        R["steps"].append([name, repr(e)[:300]]); return None

metrics = json.load(open(f"{STAGE}/metrics.json"))
R["metrics"] = metrics

step("create_model", lambda: w.registered_models.create(
    catalog_name=CAT, schema_name=SCH, name=NAME,
    comment="Fine-tuned char-level bigram LM (ftjob698f06)"))

mv = ac.do("POST", "/api/2.0/mlflow/unity-catalog/model-versions/create",
           body={"name": FULL, "source": STAGE})["model_version"]
version = int(mv["version"]); loc = mv["storage_location"]
R["version"] = version; R["storage_location"] = loc

creds = ac.do("POST", CREDS_EP,
              body={"name": FULL, "version": version,
                    "operation": "MODEL_VERSION_OPERATION_READ_WRITE"})
c = creds.get("credentials", creds)
aws = c.get("aws_temp_credentials", c)
# record redacted shape for diagnostics
R["creds_shape"] = {k: (list(v.keys()) if isinstance(v, dict) else "scalar")
                    for k, v in c.items()}

s3 = boto3.client("s3",
    aws_access_key_id=aws["access_key_id"],
    aws_secret_access_key=aws["secret_access_key"],
    aws_session_token=aws["session_token"],
    region_name="us-west-2")
bucket, _, key_prefix = loc[len("s3://"):].partition("/")
key = f"{key_prefix.rstrip('/')}/finetuned_model.npz"
with open(f"{STAGE}/finetuned_model.npz", "rb") as fh:
    s3.put_object(Bucket=bucket, Key=key, Body=fh.read())
R["uploaded_key"] = key

fin = ac.do("POST", "/api/2.0/mlflow/unity-catalog/model-versions/finalize",
            body={"name": FULL, "version": version})
R["finalized_status"] = fin.get("model_version", {}).get("status")

for k, v in metrics.items():
    step(f"tag:{k}", lambda k=k, v=v: ac.do(
        "POST", "/api/2.0/mlflow/unity-catalog/model-versions/set-tag",
        body={"name": FULL, "version": version, "key": k, "value": str(v)}))
step("alias", lambda: w.registered_models.set_alias(FULL, "prod", version))

dbutils.notebook.exit(json.dumps(R))
