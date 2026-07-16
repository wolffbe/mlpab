# Databricks notebook source
# Create synced table for low-latency access to predictions

import requests

CATALOG = "workspace"
SCHEMA = "mlpab4d3871"
PRED_TABLE = "ccpred739ee9"

host = spark.conf.get("spark.databricks.workspaceUrl", "")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
if not host.startswith("https://"):
    host = f"https://{host}"

print(f"Host: {host}")
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

# COMMAND ----------
# Try creating a synced table (replacement for deprecated online tables)
table_full_name = f"{CATALOG}.{SCHEMA}.{PRED_TABLE}"

# Method 1: Try the synced tables API
synced_spec = {
    "name": f"{table_full_name}_synced",
    "spec": {
        "source_table_full_name": table_full_name,
        "primary_key_columns": ["transaction_id"],
        "run_triggered": {"triggered_updates_schedule": None}
    }
}

resp = requests.post(f"{host}/api/2.0/synced-tables", headers=headers, json=synced_spec)
print(f"Synced table (method 1): {resp.status_code} - {resp.text[:500]}")

# COMMAND ----------
# Method 2: Try online-tables with timeseries
online_spec = {
    "name": f"{table_full_name}_online",
    "spec": {
        "source_table_full_name": table_full_name,
        "primary_key_columns": ["transaction_id"],
        "run_continuously": {}
    }
}

resp2 = requests.post(f"{host}/api/2.0/online-tables", headers=headers, json=online_spec)
print(f"Online table (method 2): {resp2.status_code} - {resp2.text[:500]}")

# COMMAND ----------
# List online tables to see what's available
resp3 = requests.get(f"{host}/api/2.0/online-tables", headers=headers)
print(f"List online tables: {resp3.status_code}")

# COMMAND ----------
print("Done. Predictions table is available as Delta table for offline/batch access.")
print(f"Table: {table_full_name}")
