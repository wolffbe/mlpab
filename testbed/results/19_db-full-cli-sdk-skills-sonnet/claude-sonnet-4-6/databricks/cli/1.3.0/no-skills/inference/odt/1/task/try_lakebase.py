# Databricks notebook source
# COMMAND ----------
import json
import requests

host = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

results = {}

# Check Lakebase/Postgres API
resp = requests.get(f"https://{host}/api/2.0/lakebase/projects", headers=headers)
results["GET /api/2.0/lakebase/projects"] = f"{resp.status_code}: {resp.text[:200]}"

resp = requests.get(f"https://{host}/api/2.0/postgres/projects", headers=headers)
results["GET /api/2.0/postgres/projects"] = f"{resp.status_code}: {resp.text[:200]}"

# Try the Databricks SDK for Synced Tables in newer version paths
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

# Check if there is any internal API client that handles synced tables
api = w.api_client
results["api_client_type"] = str(type(api))

# Try to call the synced-tables API using the raw API client
try:
    result = api.do("POST", "/api/2.0/online-tables", body={
        "name": "workspace.mlpaba35f2a.scored50223c_online",
        "spec": {
            "source_table_full_name": "workspace.mlpaba35f2a.scored50223c",
            "primary_key_columns": ["request_id"],
            "run_triggered": {}
        }
    })
    results["raw_api"] = str(result)[:300]
except Exception as e:
    results["raw_api_error"] = str(e)[:300]

# Try GET for existing online tables
resp = requests.get(f"https://{host}/api/2.0/online-tables?name=workspace.mlpaba35f2a.scored50223c_online", headers=headers)
results["GET online table with param"] = f"{resp.status_code}: {resp.text[:200]}"

dbutils.notebook.exit(json.dumps(results))
