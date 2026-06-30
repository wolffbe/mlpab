# Databricks notebook source
# MAGIC %md
# MAGIC # Create Synced Table for Real-time Access

# COMMAND ----------

import requests
import json

# Get token from dbutils
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()

print(f"Host: {host}")

# Try different API endpoints for synced/online tables
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

catalog = "workspace"
schema = "mlpab312fe6"
source_table = f"{catalog}.{schema}.incremental3526e9"
synced_table_name = f"{catalog}.{schema}.incremental3526e9_online"

# COMMAND ----------

# Try the synced tables API
payload = {
    "name": synced_table_name,
    "spec": {
        "source_table_full_name": source_table,
        "primary_key_columns": ["row_id"],
        "run_triggered": {}
    }
}

# Try different endpoints
endpoints_to_try = [
    "/api/2.0/synced-tables",
    "/api/2.1/unity-catalog/synced-tables",
    "/api/2.0/catalog/synced-tables",
    "/api/2.0/sql/synced-tables",
]

for endpoint in endpoints_to_try:
    url = f"{host}{endpoint}"
    print(f"\nTrying: POST {url}")
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        print(f"Status: {resp.status_code}")
        print(f"Response: {resp.text[:500]}")
        if resp.status_code in (200, 201):
            print("SUCCESS!")
            break
    except Exception as e:
        print(f"Error: {e}")

# COMMAND ----------

# Also try the old online-tables endpoint to see the exact error
url = f"{host}/api/2.0/online-tables"
print(f"\nTrying: POST {url}")
resp = requests.post(url, headers=headers, json=payload, timeout=30)
print(f"Status: {resp.status_code}")
print(f"Response: {resp.text[:1000]}")
