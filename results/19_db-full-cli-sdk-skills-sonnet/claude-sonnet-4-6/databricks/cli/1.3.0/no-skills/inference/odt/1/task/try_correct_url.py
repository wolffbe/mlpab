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

# Now using the correct URL - test database field at various locations
for payload_name, payload in [
    # Database at top level alongside spec
    ("toplevel_db_with_spec", {"spec": {"source_table_full_name": source}, "database": db_path}),
    # Various field name variations at top level
    ("toplevel_pg_database", {"spec": {"source_table_full_name": source}, "postgres_database": db_path}),
    ("toplevel_lakebase", {"spec": {"source_table_full_name": source}, "lakebase_database": db_path}),
    # Try numeric project reference
    ("toplevel_project", {"spec": {"source_table_full_name": source}, "project": "mlpab08bf79-ccpred"}),
    # Nested db under spec (already tried but with correct URL)
    ("spec_db", {"spec": {"source_table_full_name": source, "database": db_path}}),
    ("spec_pg_db", {"spec": {"source_table_full_name": source, "postgres_database": db_path}}),
    # Minimal variants with correct URL
    ("db_only", {"database": db_path}),
    ("spec_source_only", {"spec": {"source_table_full_name": source}}),
]:
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    msg = r.json().get("message", r.text[:200])
    results[payload_name] = f"{r.status_code}: {msg[:200]}"
    if r.status_code in (200, 201, 202):
        results["SUCCESS"] = payload_name
        break

dbutils.notebook.exit(json.dumps(results))
