# Databricks notebook source

# COMMAND ----------
import requests
import json

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

results = {}

url = f"{host}/api/2.0/database/synced_tables"

# name is the DESTINATION (in the schema), source is the Delta table
body = {
    "database_instance_name": "mlpab9e4ddc-features-online",
    "logical_database_name": "featuresdb",
    "name": "workspace.mlpab9e4ddc.featuresb1ea93_online",
    "spec": {
        "scheduling_policy": "TRIGGERED",
        "source_table_full_name": "workspace.mlpab9e4ddc.featuresb1ea93",
        "primary_key_columns": ["row_id"],
        "timeseries_key": "event_time",
    }
}

resp = requests.post(url, headers=headers, json=body, timeout=30)
results["create_synced_table"] = f"{resp.status_code}: {resp.text[:1000]}"

dbutils.notebook.exit(json.dumps(results))
