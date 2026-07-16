# Databricks notebook source
# COMMAND ----------
import json
import requests

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

results = {}
db_path = "projects/mlpab08bf79-ccpred/branches/production/databases/databricks-postgres"
catalog_id = "mlpab08bf79uccat"

# Try creating catalog with `catalog` wrapper at REST API level
# The catalog inner object needs the database reference

# First, try with catalog wrapper with different field names
for field_name, value in [
    ("database", db_path),
    ("database_full_name", db_path),
    ("lakebase_database", db_path),
    ("source_database", db_path),
    ("database_name", "databricks-postgres"),
    ("database_path", db_path),
]:
    payload = {"catalog": {field_name: value}}
    r = requests.post(
        f"{host}/api/2.0/postgres/catalogs?catalog_id={catalog_id}",
        headers=headers, json=payload, timeout=30
    )
    msg = r.json().get("message", r.text[:300])
    results[f"catalog_{field_name}"] = f"{r.status_code}: {msg[:200]}"
    if "catalog" not in msg.lower() or r.status_code in (200, 201, 202):
        results["PROGRESS"] = f"Field {field_name} works!"
        break

# Also try without catalog wrapper
for field_name, value in [
    ("database", db_path),
    ("lakebase_database", db_path),
    ("database_name", "databricks-postgres"),
]:
    payload = {field_name: value}
    r = requests.post(
        f"{host}/api/2.0/postgres/catalogs?catalog_id={catalog_id}",
        headers=headers, json=payload, timeout=30
    )
    msg = r.json().get("message", r.text[:300])
    results[f"nowrap_{field_name}"] = f"{r.status_code}: {msg[:200]}"

dbutils.notebook.exit(json.dumps(results))
