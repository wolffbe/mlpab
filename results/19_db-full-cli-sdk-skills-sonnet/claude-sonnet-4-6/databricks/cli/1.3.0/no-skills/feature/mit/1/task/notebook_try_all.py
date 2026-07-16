# Databricks notebook source
# COMMAND ----------
import json, requests

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()

results = {}

# Try SQL-based synced table creation via various approaches
sql_statements = [
    # New Databricks SQL syntax for creating a synced/online table
    "CREATE OR REPLACE TABLE workspace.mlpabf1452c.featuresb1ea93_online USING SYNCED FROM workspace.mlpabf1452c.featuresb1ea93",
    "CREATE TABLE workspace.mlpabf1452c.featuresb1ea93_online SYNCED FROM workspace.mlpabf1452c.featuresb1ea93 INDEXED BY row_id",
    "CREATE ONLINE TABLE workspace.mlpabf1452c.featuresb1ea93_online AS SELECT * FROM workspace.mlpabf1452c.featuresb1ea93",
    # Try the spark create table with special options
    "CREATE TABLE IF NOT EXISTS workspace.mlpabf1452c.featuresb1ea93_online USING delta TBLPROPERTIES ('delta.enableDatabricksOnlineTable' = 'true') AS SELECT * FROM workspace.mlpabf1452c.featuresb1ea93",
]

for stmt in sql_statements:
    try:
        r = requests.post(
            f"{host}/api/2.0/sql/statements",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"warehouse_id": "4dfab06c923fe3cc", "statement": stmt, "wait_timeout": "30s"},
            timeout=35
        )
        resp = r.json()
        results[stmt[:60]] = {"status": r.status_code, "state": resp.get("status", {}).get("state"), "error": resp.get("status", {}).get("error")}
    except Exception as e:
        results[stmt[:60]] = {"error": str(e)}

# Try directly via REST API with the proper request format
try:
    # The new Databricks synced-tables API
    r = requests.post(
        f"{host}/api/2.0/online-tables",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={
            "name": "workspace.mlpabf1452c.featuresb1ea93_online",
            "spec": {
                "source_table_full_name": "workspace.mlpabf1452c.featuresb1ea93",
                "primary_key_columns": ["row_id"],
                "timeseries_key": "event_time",
                "run_triggered": {
                    "triggered": True
                }
            }
        },
        timeout=30
    )
    results['online_tables_api'] = {"status": r.status_code, "body": r.text[:300]}
except Exception as e:
    results['online_tables_api'] = {"error": str(e)}

dbutils.notebook.exit(json.dumps(results))
