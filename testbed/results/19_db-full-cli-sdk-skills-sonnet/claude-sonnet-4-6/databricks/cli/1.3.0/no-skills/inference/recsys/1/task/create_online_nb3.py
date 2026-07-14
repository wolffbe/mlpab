# Databricks notebook source
# COMMAND ----------
import requests
import json

host = spark.conf.get("spark.databricks.workspaceUrl", "")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

results = {}

# Test multiple paths
paths_to_test = [
    ("GET", "/api/2.0/online-tables"),
    ("GET", "/api/2.1/online-tables"),
    ("GET", "/api/2.0/preview/synced-tables"),
    ("GET", "/api/2.1/preview/synced-tables"),
    ("GET", "/api/2.0/serving-endpoints"),
]

for method, path in paths_to_test:
    resp = requests.get(f"https://{host}{path}", headers=headers)
    results[f"{method} {path}"] = f"{resp.status_code}: {resp.text[:100]}"

# Try to list online tables
resp = requests.get(f"https://{host}/api/2.1/unity-catalog/online-tables", headers=headers)
results["GET /api/2.1/unity-catalog/online-tables"] = f"{resp.status_code}: {resp.text[:200]}"

# Now try creating synced table with different API endpoints
table_name = "workspace.mlpabb40f43.recs708df6"
payload = {
    "name": "workspace.mlpabb40f43.recs708df6_online",
    "spec": {
        "source_table_full_name": table_name,
        "primary_key_columns": ["rec_id"],
        "run_triggered": {}
    }
}

post_paths = [
    "/api/2.1/unity-catalog/online-tables",
    "/api/2.0/preview/online-tables",
]

for path in post_paths:
    resp = requests.post(f"https://{host}{path}", headers=headers, json=payload)
    results[f"POST {path}"] = f"{resp.status_code}: {resp.text[:300]}"

dbutils.notebook.exit(json.dumps(results, indent=2))
