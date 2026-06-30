# Databricks notebook source
# COMMAND ----------
import json, requests

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()

results = {}

DB_PATH = "projects/mlpabf1452c-feat/branches/production/databases/databricks-postgres"
DB_ID = "databricks-postgres"
SOURCE = "workspace.mlpabf1452c.featuresb1ea93"
TARGET = "workspace.mlpabf1452c.featuresb1ea93_online"

# The server says "Must specify database field" but none of the spec sub-fields work
# Try database at the TOP level (outside of spec)

variations = [
    # TOP-level database fields
    {"spec": {"source_table_full_name": SOURCE}, "database": DB_PATH},
    {"spec": {"source_table_full_name": SOURCE}, "database": DB_ID},
    {"spec": {"source_table_full_name": SOURCE}, "postgres_database": DB_PATH},
    {"spec": {"source_table_full_name": SOURCE}, "pg_database": DB_PATH},
    # Maybe "database" is a nested object
    {"spec": {"source_table_full_name": SOURCE}, "database": {"name": DB_PATH}},
    {"spec": {"source_table_full_name": SOURCE}, "database": {"path": DB_PATH}},
    # Try the catalog_name as the "database" field
    {"spec": {"source_table_full_name": SOURCE}, "catalog_name": "workspace"},
    {"spec": {"source_table_full_name": SOURCE}, "schema_name": "mlpabf1452c"},
    # Try the UC schema notation
    {"spec": {"source_table_full_name": SOURCE}, "schema": "workspace.mlpabf1452c"},
]

for i, payload in enumerate(variations):
    try:
        r = requests.post(
            f"{host}/api/2.0/postgres/synced_tables?synced_table_id={TARGET}",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        results[f'v{i}'] = {"status": r.status_code, "body": r.text[:400]}
        if r.status_code in (200, 201):
            break
    except Exception as e:
        results[f'v{i}'] = {"error": str(e)}

dbutils.notebook.exit(json.dumps(results))
