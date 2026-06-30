# Databricks notebook source
import importlib, json
result = {}
out = {}
for m in ["boto3", "mlflow", "numpy"]:
    try:
        importlib.import_module(m); out[m] = "OK"
    except Exception as e:
        out[m] = f"NO:{type(e).__name__}"
result["mods"] = out

# Can we issue temp path credentials from serverless (Databricks compute)?
try:
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.catalog import PathOperation
    w = WorkspaceClient()
    full = "workspace.mlpab5b087a.ftmodel698f06"
    mv = w.model_versions.get(full, 2)
    r = w.temporary_path_credentials.generate_temporary_path_credentials(
        url=mv.storage_location, operation=PathOperation.PATH_READ_WRITE)
    d = r.as_dict()
    result["credshape"] = {k: (list(v.keys()) if isinstance(v, dict) else str(v)[:40]) for k, v in d.items()}
except Exception as e:
    result["crederr"] = repr(e)[:400]

dbutils.notebook.exit(json.dumps(result))
