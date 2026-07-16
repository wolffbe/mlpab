# Databricks notebook source
# COMMAND ----------
import requests
import json

host = spark.conf.get("spark.databricks.workspaceUrl", "")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

results = {}

# Try different field names for the source table in synced_tables spec
field_names = [
    "source_uc_table_name",
    "delta_table_name",
    "source_delta_table",
    "uc_table_name",
    "source_table",
    "table",
    "name",
    "from_table",
    "origin_table",
]

table_name = "workspace.mlpabb40f43.recs708df6"

for field in field_names:
    payload = {
        "name": "workspace.mlpabb40f43.recs708df6_online",
        "database_instance_name": "mlpabb40f43-db",
        "spec": {
            field: table_name,
            "scheduling_policy": "TRIGGERED"
        }
    }
    resp = requests.post(f"https://{host}/api/2.0/database/synced_tables", headers=headers, json=payload)
    results[f"spec.{field}"] = f"{resp.status_code}: {resp.text[:200]}"

dbutils.notebook.exit(json.dumps(results, indent=2))
