# Databricks notebook source
# COMMAND ----------
import json
import requests

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

results = {}
db_path = "projects/mlpab08bf79-ccpred/branches/production/databases/databricks-postgres"
source = "workspace.mlpaba35f2a.scored50223c"
catalog_id = "mlpab08bf79uccat"

# Step 1: Try to create a Lakebase UC catalog via REST API
# Try with different field names for the database reference
for field_name, field_value in [
    ("database", db_path),
    ("database_full_name", db_path),
    ("lakebase_database", db_path),
    ("parent", db_path),
    ("parent_database", db_path),
    ("database_name", "databricks-postgres"),
    ("source", db_path),
]:
    payload = {field_name: field_value}
    r = requests.post(
        f"{host}/api/2.0/postgres/catalogs?catalog_id={catalog_id}",
        headers=headers, json=payload, timeout=30
    )
    msg = r.json().get("message", r.text[:200])
    results[f"catalog_{field_name}"] = f"{r.status_code}: {msg[:200]}"
    if "catalog" not in msg.lower() or r.status_code in (200, 201, 202):
        results[f"PROGRESS_catalog_{field_name}"] = True
        break

# Step 2: Try creating synced table with Lakebase catalog path
# (assuming catalog was created or already exists)
for st_id in [
    f"{catalog_id}.mlpaba35f2a.scored50223c",
    f"{catalog_id}.mlpaba35f2a.scored50223c_online",
]:
    payload = {"spec": {"source_table_full_name": source}}
    r = requests.post(
        f"{host}/api/2.0/postgres/synced_tables?synced_table_id={st_id}",
        headers=headers, json=payload, timeout=30
    )
    msg = r.json().get("message", r.text[:200])
    results[f"synced_{st_id}"] = f"{r.status_code}: {msg[:200]}"

dbutils.notebook.exit(json.dumps(results))
