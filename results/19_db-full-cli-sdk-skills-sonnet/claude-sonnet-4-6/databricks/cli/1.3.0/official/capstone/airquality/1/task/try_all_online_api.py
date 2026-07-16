# Databricks notebook source

# COMMAND ----------
import json
results = {}

from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

full_schema = "workspace.mlpab375647"
pred_name = "airqpredfdfb59"
online_table_name = f"{full_schema}.{pred_name}_online"

# Try all possible endpoint/method combinations for Synced Tables
tests = {
    "POST /api/2.0/synced-tables": {
        "method": "POST",
        "path": "/api/2.0/synced-tables",
        "body": {
            "name": online_table_name,
            "spec": {
                "source_table_full_name": f"{full_schema}.{pred_name}",
                "primary_key_columns": ["date"],
                "run_triggered": {}
            }
        }
    },
    "POST /api/2.1/online-tables": {
        "method": "POST",
        "path": "/api/2.1/online-tables",
        "body": {
            "name": online_table_name,
            "spec": {
                "source_table_full_name": f"{full_schema}.{pred_name}",
                "primary_key_columns": ["date"],
                "run_triggered": {}
            }
        }
    },
    "POST /api/2.0/online-tables (v2)": {
        "method": "POST",
        "path": "/api/2.0/online-tables",
        "body": {
            "table": {
                "name": online_table_name,
                "spec": {
                    "source_table_full_name": f"{full_schema}.{pred_name}",
                    "primary_key_columns": [{"name": "date"}],
                    "run_triggered": {"triggered_updates": {}}
                }
            }
        }
    }
}

for name, test in tests.items():
    try:
        resp = w.api_client.do(test["method"], test["path"], body=test["body"])
        results[name] = "OK: " + json.dumps(resp)[:200]
    except Exception as e:
        results[name] = f"ERR({type(e).__name__}): {str(e)[:200]}"

dbutils.notebook.exit(json.dumps(results))
