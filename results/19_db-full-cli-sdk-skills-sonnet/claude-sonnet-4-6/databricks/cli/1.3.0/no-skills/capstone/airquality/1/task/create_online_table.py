# Databricks notebook source

schema = "workspace.mlpabd7768b"
pred_name = "airqpredfdfb59"

# COMMAND ----------

# Try to create a synced table for low-latency serving
import requests

ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
token = ctx.apiToken().get()
host = ctx.apiUrl().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

print(f"Host: {host}")

# Try synced tables endpoint
spec = {
    "name": f"{schema}.{pred_name}_synced",
    "spec": {
        "source_table_full_name": f"{schema}.{pred_name}",
        "primary_key_columns": ["date"],
        "run_triggered": {}
    }
}

for api_path in ["/api/2.0/synced-tables", "/api/2.1/synced-tables", "/api/2.0/feature-serving/feature-tables"]:
    resp = requests.post(f"{host}{api_path}", headers=headers, json=spec)
    print(f"{api_path}: {resp.status_code} - {resp.text[:200]}")

# COMMAND ----------

# Check all available API routes for online/synced tables
for method, path in [
    ("GET", "/api/2.0/online-tables"),
    ("GET", "/api/2.0/synced-tables"),
    ("GET", "/api/2.0/serving-endpoints"),
    ("GET", "/api/2.0/feature-store/feature-tables"),
]:
    resp = requests.request(method, f"{host}{path}", headers=headers)
    print(f"{method} {path}: {resp.status_code}")

# COMMAND ----------

# Try creating via catalog API (Synced Tables are managed via UC catalog)
catalog_spec = {
    "name": f"workspace.mlpabd7768b.{pred_name}_synced",
    "spec": {
        "source_table_full_name": f"workspace.mlpabd7768b.{pred_name}",
        "primary_key_columns": ["date"],
        "run_triggered": {"triggered_enable_continuous": False}
    }
}

resp = requests.post(f"{host}/api/2.0/online-tables", headers=headers, json={"name": f"workspace.mlpabd7768b.{pred_name}_online", "spec": {"source_table_full_name": f"workspace.mlpabd7768b.{pred_name}", "primary_key_columns": ["date"], "run_triggered": {}}})
print(f"Online table: {resp.status_code} - {resp.text[:500]}")

# COMMAND ----------
print("Done")
