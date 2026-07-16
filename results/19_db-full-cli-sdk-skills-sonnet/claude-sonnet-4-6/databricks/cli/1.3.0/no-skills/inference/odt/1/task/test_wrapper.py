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

# Test with synced_table wrapper (CLI-style)
for payload_name, payload in [
    # Wrapped in synced_table like the CLI does
    ("wrapped_spec_source", {"synced_table": {"spec": {"source_table_full_name": source}}}),
    # With database inside spec but wrapped
    ("wrapped_spec_source_db", {"synced_table": {"spec": {"source_table_full_name": source, "database": db_path}}}),
    # With database at synced_table level
    ("wrapped_db_toplevel", {"synced_table": {"spec": {"source_table_full_name": source}, "database": db_path}}),
    # Just spec.source with no wrapper (to verify server behavior)
    ("direct_spec", {"spec": {"source_table_full_name": source}}),
    # Wrapped with database at different levels
    ("wrapped_only_db", {"synced_table": {"database": db_path, "spec": {"source_table_full_name": source}}}),
    # Try with database as the full resource name without spec
    ("wrapped_full_name", {"synced_table": {"name": source, "database": db_path}}),
]:
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    msg = r.json().get("message", r.text[:100])
    results[payload_name] = f"{r.status_code}: {msg[:200]}"

dbutils.notebook.exit(json.dumps(results))
