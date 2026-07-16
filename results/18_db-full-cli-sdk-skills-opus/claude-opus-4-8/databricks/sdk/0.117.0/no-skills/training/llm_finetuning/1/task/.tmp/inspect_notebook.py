# Databricks notebook source
import json
from databricks.sdk import WorkspaceClient
w = WorkspaceClient(); ac = w.api_client
FULL = "workspace.mlpab5b087a.ftmodel698f06"
creds = ac.do("POST", "/api/2.0/mlflow/unity-catalog/model-versions/generate-temporary-credentials",
              body={"name": FULL, "version": 1, "operation": "MODEL_VERSION_OPERATION_READ_WRITE"})


def redact(o, depth=0):
    if isinstance(o, dict):
        out = {}
        for k, v in o.items():
            if isinstance(v, (dict, list)):
                out[k] = redact(v, depth + 1)
            elif any(s in k.lower() for s in ["secret", "session", "token", "access_key"]):
                out[k] = f"<{type(v).__name__}:{len(str(v))}>"
            else:
                out[k] = v
        return out
    if isinstance(o, list):
        return [redact(x, depth + 1) for x in o]
    return o

dbutils.notebook.exit(json.dumps(redact(creds)))
