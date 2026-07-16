# Databricks notebook source
# COMMAND ----------
import json
import requests

host = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

results = {}

# Try the Postgres synced tables API
for method, path, data in [
    ("GET", "/api/2.0/postgres/synced-tables", None),
    ("GET", "/api/2.0/postgres/synced_tables", None),
    ("POST", "/api/2.0/postgres/synced-tables", {"synced_table_id": "workspace.mlpaba35f2a.scored50223c"}),
    ("POST", "/api/2.0/postgres/synced_tables/workspace.mlpaba35f2a.scored50223c", {}),
]:
    if method == "GET":
        resp = requests.get(f"https://{host}{path}", headers=headers)
    else:
        resp = requests.post(f"https://{host}{path}", json=data, headers=headers)
    results[f"{method} {path}"] = f"{resp.status_code}: {resp.text[:200]}"

# Try to use the SDK's postgres client
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
pg_attrs = [a for a in dir(w) if "postgres" in a.lower() or "lakebase" in a.lower()]
results["sdk_pg_attrs"] = pg_attrs

# Check if w.postgres exists and what it can do
if hasattr(w, 'postgres'):
    pg = w.postgres
    results["pg_methods"] = [m for m in dir(pg) if not m.startswith('_')]

dbutils.notebook.exit(json.dumps(results))
