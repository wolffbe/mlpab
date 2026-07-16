# Databricks notebook source
# COMMAND ----------
import json, requests

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()

results = {}

DB_PATH = "projects/mlpabf1452c-feat/branches/production/databases/databricks-postgres"
DB_ID = "databricks-postgres"
DB_PG = "databricks_postgres"
SOURCE = "workspace.mlpabf1452c.featuresb1ea93"
TARGET = "workspace.mlpabf1452c.featuresb1ea93_online"

# The CLI sends spec.postgres_database to the server
# Let's figure out what value format the server accepts for "database"
# by testing all possible values

variations = [
    # Different values for postgres_database
    {"spec": {"source_table_full_name": SOURCE, "postgres_database": DB_PATH}},
    {"spec": {"source_table_full_name": SOURCE, "postgres_database": DB_ID}},
    {"spec": {"source_table_full_name": SOURCE, "postgres_database": DB_PG}},
    # Maybe the field name in JSON IS "database" (not postgres_database)
    {"spec": {"source_table_full_name": SOURCE, "database": DB_PATH}},
    {"spec": {"source_table_full_name": SOURCE, "database": DB_ID}},
    {"spec": {"source_table_full_name": SOURCE, "database": DB_PG}},
    # Maybe database is the full databases/ path
    {"spec": {"source_table_full_name": SOURCE, "database": f"{DB_PATH}"}},
    # Try with primary_key_columns as well
    {"spec": {"source_table_full_name": SOURCE, "postgres_database": DB_PATH, "primary_key_columns": ["row_id"]}},
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
