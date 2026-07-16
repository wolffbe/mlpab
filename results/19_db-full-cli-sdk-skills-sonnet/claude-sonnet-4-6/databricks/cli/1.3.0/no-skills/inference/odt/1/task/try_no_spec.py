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
url = f"{host}/api/2.0/postgres/synced_tables?synced_table_id={source}"

# Try without any spec wrapper
payloads = {
    "no_spec_flat": {"source_table_full_name": source, "database": db_path},
    "no_spec_db_only": {"database": db_path},
    "source_only": {"source_table_full_name": source},
    # Try with different top-level structure
    "config_wrapper": {"config": {"source_table_full_name": source, "database": db_path}},
    "table_wrapper": {"table": {"source_table_full_name": source, "database": db_path}},
    "settings_wrapper": {"settings": {"source_table_full_name": source, "database": db_path}},
    # Very minimal
    "db_only_flat": {"database": db_path},
    # Try with explicit "synced_table" containing "source_table_full_name" at top level
    "hybrid": {"source_table_full_name": source, "spec": {"database": db_path}},
}

for name, payload in payloads.items():
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    msg = r.json().get("message", r.text[:100])
    results[name] = f"{r.status_code}: {msg[:200]}"

dbutils.notebook.exit(json.dumps(results))
