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

# Without wrapper, source is recognized. Try every variant for the database field
# in the spec object (direct body, no wrapper)
candidates = [
    # Standard naming variations
    "database", "postgres_database", "lakebase_database", "pg_database",
    "database_full_name", "database_resource_name", "database_path", "database_name",
    "target_database", "pg_db", "lakebase_db", "db", "database_id",
    # Possible nested paths
    "db_resource", "pg_catalog", "postgres_catalog",
    # With parent reference
    "parent", "parent_database", "db_parent",
    # Catalog-based
    "catalog_database", "lakebase_catalog",
]

for field_name in candidates:
    payload = {"spec": {"source_table_full_name": source, field_name: db_path}}
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    msg = r.json().get("message", r.text[:100])
    if "database" not in msg.lower():
        # Different error means field was recognized!
        results[f"RECOGNIZED_{field_name}"] = f"{r.status_code}: {msg[:300]}"
    results[field_name] = f"{r.status_code}: {msg[:80]}"

# Also try: just "database" but as an absolute Postgres project (not branch path)
for db_value in [
    "mlpab08bf79-ccpred",
    "databricks-postgres",
    "databricks_postgres",
    "projects/mlpab08bf79-ccpred",
    "projects/mlpab08bf79-ccpred/branches/production",
]:
    payload = {"spec": {"source_table_full_name": source, "database": db_value}}
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    msg = r.json().get("message", r.text[:100])
    results[f"db_val_{db_value[:20]}"] = f"{r.status_code}: {msg[:80]}"

dbutils.notebook.exit(json.dumps(results))
