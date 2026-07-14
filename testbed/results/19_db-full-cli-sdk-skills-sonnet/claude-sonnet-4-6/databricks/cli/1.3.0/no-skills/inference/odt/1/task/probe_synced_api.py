# Databricks notebook source
# COMMAND ----------
import json
import requests

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

results = {}

# Try using protobuf JSON format with camelCase
# The Databricks postgres API uses protobuf-based JSON
# Try with "database" at spec level using different encodings
db_path = "projects/mlpab08bf79-ccpred/branches/production/databases/databricks-postgres"
source = "workspace.mlpaba35f2a.scored50223c"

# What if the spec uses camelCase?
for payload in [
    {"spec": {"sourceTableFullName": source, "database": db_path}},
    {"spec": {"sourceTableFullName": source}, "database": db_path},
    {"spec": {"sourceTableFullName": source, "postgresDatabase": db_path}},
    {"spec": {"sourceTableFullName": source, "lakebaseDatabase": db_path}},
    {"spec": {"sourceTableFullName": source, "postgresDb": db_path}},
    # What if the synced_table_id in query param is wrong format?
    # Try the schema.table without catalog
    {"spec": {"sourceTableFullName": source}},
]:
    r = requests.post(
        f"{host}/api/2.0/postgres/synced_tables?synced_table_id={source}",
        headers=headers, json=payload, timeout=30
    )
    key = str(sorted(payload.items()))[:60]
    results[key] = f"{r.status_code}: {r.json().get('message', r.text[:100])}"

# Also try GET for the existing postgres projects catalogs
r = requests.get(f"{host}/api/2.0/postgres/catalogs", headers=headers, timeout=30)
results["GET_catalogs"] = f"{r.status_code}: {r.text[:300]}"

# Check if there are any existing synced tables in ANY schema
for schema in ["mlpab229d43", "mlpabf1452c", "mlpab5c18ba"]:
    r = requests.get(
        f"{host}/api/2.0/postgres/synced_tables?synced_table_id=workspace.{schema}.test",
        headers=headers, timeout=30
    )
    results[f"get_{schema}"] = f"{r.status_code}: {r.text[:100]}"

dbutils.notebook.exit(json.dumps(results))
