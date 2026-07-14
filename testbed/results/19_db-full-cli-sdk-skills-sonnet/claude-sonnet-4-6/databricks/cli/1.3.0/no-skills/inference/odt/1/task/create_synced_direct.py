# Databricks notebook source
# COMMAND ----------
import json
import requests

host = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

results = {}

db_path = "projects/mlpab08bf79-ccpred/branches/production/databases/databricks-postgres"
base_url = f"https://{host}/api/2.0/postgres/synced_tables"

# Try the REST API directly with `database` field - bypass CLI validation
body = {
    "spec": {
        "source_table_full_name": "workspace.mlpaba35f2a.scored50223c",
        "database": db_path
    }
}
resp = requests.post(
    f"{base_url}?synced_table_id=workspace.mlpaba35f2a.scored50223c",
    json=body,
    headers=headers
)
results["with_database"] = f"{resp.status_code}: {resp.text[:400]}"

# Also try with database at top level (not in spec)
body2 = {
    "spec": {"source_table_full_name": "workspace.mlpaba35f2a.scored50223c"},
    "database": db_path
}
resp2 = requests.post(
    f"{base_url}?synced_table_id=workspace.mlpaba35f2a.scored50223c",
    json=body2,
    headers=headers
)
results["database_toplevel"] = f"{resp2.status_code}: {resp2.text[:400]}"

# Try with different key variations
for key in ["database", "lakebase_database", "target_database", "db", "database_resource"]:
    body3 = {"spec": {"source_table_full_name": "workspace.mlpaba35f2a.scored50223c"}, key: db_path}
    r = requests.post(f"{base_url}?synced_table_id=workspace.mlpaba35f2a.scored50223c", json=body3, headers=headers)
    msg = r.json().get("message", "")
    results[f"toplevel_{key}"] = f"{r.status_code}: {msg[:100]}"

dbutils.notebook.exit(json.dumps(results))
