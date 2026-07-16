# Databricks notebook source
import json
import requests

ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
token = ctx.apiToken().get()
host = ctx.apiUrl().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

schema = "workspace.mlpabcef85c"
table_name = "profilesaa70e4"
full_table_name = f"{schema}.{table_name}"

results = {}

# Check multiple API paths
paths = [
    ("GET", "/api/2.0/synced-tables", {}),
    ("GET", "/api/2.1/synced-tables", {}),
    ("GET", "/api/2.0/synced_tables", {}),
    ("POST", "/api/2.0/synced-tables", {"name": f"{full_table_name}_st", "spec": {"source_table_full_name": full_table_name, "primary_key_columns": ["account_id"], "run_triggered": {}}}),
    ("POST", "/api/2.1/synced-tables", {"name": f"{full_table_name}_st", "spec": {"source_table_full_name": full_table_name, "primary_key_columns": ["account_id"], "run_triggered": {}}}),
    ("GET", f"/api/2.0/synced-tables/{full_table_name}_st", {}),
    ("GET", f"/api/2.1/synced-tables/{full_table_name}_st", {}),
]

for method, path, body in paths:
    try:
        if method == "GET":
            r = requests.get(f"{host}{path}", headers=headers)
        else:
            r = requests.post(f"{host}{path}", headers=headers, json=body)
        results[f"{method} {path}"] = {"status": r.status_code, "text": r.text[:300]}
    except Exception as e:
        results[f"{method} {path}"] = {"error": str(e)}

print(json.dumps(results, indent=2))
dbutils.notebook.exit(json.dumps(results))
