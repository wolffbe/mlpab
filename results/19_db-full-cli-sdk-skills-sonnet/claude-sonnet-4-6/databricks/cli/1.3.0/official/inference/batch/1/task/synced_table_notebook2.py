# Databricks notebook source
# COMMAND ----------
import requests
import os

# Get the Databricks host and token from environment
host = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

# Try to create a synced table via REST API
url = f"https://{host}/api/2.0/synced-tables"
payload = {
    "name": "workspace.mlpab6ef9cb.scores4f5893_synced",
    "spec": {
        "source_table_full_name": "workspace.mlpab6ef9cb.scores4f5893",
        "primary_key_columns": ["account_id"],
        "run_triggered": {}
    }
}

response = requests.post(url, json=payload, headers=headers)
print(f"Status: {response.status_code}")
print(f"Response: {response.text[:2000]}")
