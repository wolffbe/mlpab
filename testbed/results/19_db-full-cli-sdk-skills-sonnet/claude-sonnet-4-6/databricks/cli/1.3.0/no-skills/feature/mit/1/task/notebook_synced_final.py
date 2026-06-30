# Databricks notebook source
# COMMAND ----------
import json, requests

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()

results = {}

DB_PATH = "projects/mlpabf1452c-feat/branches/production/databases/databricks-postgres"
SOURCE = "workspace.mlpabf1452c.featuresb1ea93"
TARGET = "workspace.mlpabf1452c.featuresb1ea93_online"

# Try many variations of the "database" field at different levels and with different names
payloads = [
    # Top-level "database" field
    {"synced_table": {"spec": {"source_table_full_name": SOURCE, "postgres_database": DB_PATH}, "database": DB_PATH}},
    # Inside synced_table but as database
    {"synced_table": {"spec": {"source_table_full_name": SOURCE}, "database": DB_PATH}},
    # spec only with different names
    {"spec": {"source_table_full_name": SOURCE, "database": DB_PATH}},
    {"spec": {"source_table_full_name": SOURCE, "database": "databricks-postgres"}},
    {"spec": {"source_table_full_name": SOURCE, "database_path": DB_PATH}},
    # Source table and database at top level inside synced_table
    {"synced_table": {"source_table_full_name": SOURCE, "database": DB_PATH}},
    # Try just database
    {"synced_table": {"database": DB_PATH}},
    {"database": DB_PATH, "source_table_full_name": SOURCE},
]

for i, payload in enumerate(payloads):
    try:
        r = requests.post(
            f"{host}/api/2.0/postgres/synced_tables?synced_table_id={TARGET}",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        results[f'try_{i}'] = {"status": r.status_code, "body": r.text[:400]}
        if r.status_code in (200, 201):
            break
    except Exception as e:
        results[f'try_{i}'] = {"error": str(e)}

dbutils.notebook.exit(json.dumps(results))
