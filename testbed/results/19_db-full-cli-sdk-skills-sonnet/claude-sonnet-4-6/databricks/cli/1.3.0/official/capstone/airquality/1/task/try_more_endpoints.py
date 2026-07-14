# Databricks notebook source

# COMMAND ----------
import json
results = {}

from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

full_schema = "workspace.mlpab375647"
pred_name = "airqpredfdfb59"
online_table_name = f"{full_schema}.{pred_name}_online"

body = {
    "name": online_table_name,
    "spec": {
        "source_table_full_name": f"{full_schema}.{pred_name}",
        "primary_key_columns": ["date"],
        "run_triggered": {}
    }
}

more_endpoints = [
    ("POST", "/api/2.0/preview/online-tables"),
    ("POST", "/api/2.0/catalog/online-tables"),
    ("GET", "/api/2.0/online-tables"),
    ("GET", "/api/2.0/preview/online-tables"),
    ("GET", "/api/2.0/deltasync/synced-tables"),
    ("POST", "/api/2.0/deltasync/synced-tables"),
]

for method, path in more_endpoints:
    try:
        resp = w.api_client.do(method, path, body=body if method == "POST" else None)
        results[f"{method} {path}"] = "OK: " + json.dumps(resp)[:200]
    except Exception as e:
        results[f"{method} {path}"] = f"ERR({type(e).__name__}): {str(e)[:200]}"

dbutils.notebook.exit(json.dumps(results))
