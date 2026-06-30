# Databricks notebook source
# Try to create online access for the feature table using Feature Engineering SDK

# COMMAND ----------
# First check what's available
import subprocess
result = subprocess.run(["pip", "list"], capture_output=True, text=True)
fe_lines = [l for l in result.stdout.split('\n') if 'feature' in l.lower() or 'databricks' in l.lower()]
print('\n'.join(fe_lines[:20]))

# COMMAND ----------
# Try using the Feature Engineering client to register the feature table
try:
    from databricks.feature_engineering import FeatureEngineeringClient
    fe = FeatureEngineeringClient()
    print("Feature Engineering client available")

    # Try to create online table spec
    from databricks.feature_engineering import FeatureLookup
    print("FeatureLookup available")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")

# COMMAND ----------
# Try the feature store module
try:
    from databricks import feature_store
    fs = feature_store.FeatureStoreClient()
    print("FeatureStoreClient available")
except Exception as e:
    print(f"FeatureStoreClient error: {e}")

# COMMAND ----------
# Try using databricks SDK to create online table
try:
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()
    print("WorkspaceClient available")

    # Check online tables
    print(dir(w))
except Exception as e:
    print(f"SDK error: {e}")

# COMMAND ----------
# Check available REST API paths for online feature tables
import requests
import json

# Use the cluster's token
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()

print(f"Host: {host}")

# Try synced tables endpoint
url = f"{host}/api/2.1/unity-catalog/synced-tables"
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

# List existing synced tables
resp = requests.get(url, headers=headers)
print(f"GET synced-tables: {resp.status_code}")
if resp.status_code == 200:
    print(resp.json())
else:
    print(resp.text[:500])

# COMMAND ----------
# Try to create a synced table
payload = {
    "name": "workspace.mlpab9e4ddc.featuresb1ea93_online",
    "spec": {
        "source_table_full_name": "workspace.mlpab9e4ddc.featuresb1ea93",
        "primary_key_columns": ["row_id"],
        "timeseries_key": "event_time",
        "run_triggered": {}
    }
}

resp = requests.post(url, headers=headers, json=payload)
print(f"POST synced-tables: {resp.status_code}")
print(resp.text[:1000])
