# Databricks notebook source
# COMMAND ----------
import json, requests

# Try REST API approaches for synced tables
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()

results = {}

# Test various synced table API paths
paths = [
    "/api/2.0/synced-tables",
    "/api/2.1/synced-tables",
    "/api/2.0/tables/synced",
    "/api/2.0/synced_tables",
]

for path in paths:
    try:
        r = requests.post(
            f"{host}{path}",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={
                "name": "workspace.mlpabf1452c.featuresb1ea93_online",
                "spec": {
                    "source_table_full_name": "workspace.mlpabf1452c.featuresb1ea93",
                    "primary_key_columns": ["row_id"],
                    "timeseries_key": "event_time",
                    "run_triggered": {}
                }
            },
            timeout=30
        )
        results[path] = {"status": r.status_code, "body": r.text[:300]}
    except Exception as e:
        results[path] = {"error": str(e)}

dbutils.notebook.exit(json.dumps(results))
