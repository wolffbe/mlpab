# Databricks notebook source
# COMMAND ----------
import json
import requests

host = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

results = {}

db_path = "projects/mlpab08bf79-ccpred/branches/production/databases/databricks-postgres"
base_url = f"https://{host}/api/2.0/postgres/synced_tables?synced_table_id=workspace.mlpaba35f2a.scored50223c"

# Try each database field name inside spec
for db_field in ["pg_database", "postgres_database", "lakebase_database", "parent_database",
                  "database_full_name", "db_path", "database_path", "database_id", "db_resource_name"]:
    body = {"spec": {"source_table_full_name": "workspace.mlpaba35f2a.scored50223c", db_field: db_path}}
    resp = requests.post(base_url, json=body, headers=headers)
    msg = resp.json().get("message", "")
    results[db_field] = f"{resp.status_code}: {msg[:100]}"
    if "database" not in msg.lower() or "specify" not in msg.lower():
        results[f"MAYBE_{db_field}"] = f"{resp.status_code}: {msg[:200]}"

dbutils.notebook.exit(json.dumps(results))
