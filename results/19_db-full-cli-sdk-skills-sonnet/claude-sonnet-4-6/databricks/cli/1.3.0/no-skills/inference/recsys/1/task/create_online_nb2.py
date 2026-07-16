# Databricks notebook source
# COMMAND ----------
# Explore Feature Engineering client and online store options
import requests
import json

host = spark.conf.get("spark.databricks.workspaceUrl", "")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

print(f"Workspace host: {host}")

# COMMAND ----------
# Try synced tables REST API paths
synced_paths = [
    "GET /api/2.0/online-tables",
    "GET /api/2.1/online-tables",
    "GET /api/2.0/unity-catalog/online-tables",
    "GET /api/2.1/unity-catalog/online-tables",
    "GET /api/2.0/synced-tables",
    "GET /api/2.1/synced-tables",
]

for path_str in synced_paths:
    method, path = path_str.split(" ", 1)
    try:
        resp = requests.get(f"https://{host}{path}", headers=headers)
        print(f"{path_str}: {resp.status_code} - {resp.text[:100]}")
    except Exception as e:
        print(f"{path_str}: ERROR - {e}")

# COMMAND ----------
# Try POST for synced table creation
table_name = "workspace.mlpabb40f43.recs708df6"
online_table_name = "workspace.mlpabb40f43.recs708df6_online"

# Try the API path that works
payload = {
    "name": online_table_name,
    "spec": {
        "source_table_full_name": table_name,
        "primary_key_columns": ["rec_id"],
        "run_triggered": {}
    }
}

for post_path in ["/api/2.0/online-tables", "/api/2.1/online-tables"]:
    resp = requests.post(f"https://{host}{post_path}", headers=headers, json=payload)
    print(f"POST {post_path}: {resp.status_code} - {resp.text[:300]}")

# COMMAND ----------
# Check Feature Engineering availability
try:
    from databricks.feature_engineering import FeatureEngineeringClient
    from databricks.feature_engineering import utils

    fe = FeatureEngineeringClient()
    print(f"Feature Engineering client initialized: {type(fe)}")
    print(f"Methods: {[m for m in dir(fe) if not m.startswith('_')]}")
except Exception as e:
    print(f"Error: {e}")

# COMMAND ----------
# Try the feature_store older client too
try:
    from databricks import feature_store
    fs = feature_store.FeatureStoreClient()
    print(f"Feature Store client methods: {[m for m in dir(fs) if not m.startswith('_')]}")
except Exception as e:
    print(f"Feature Store error: {e}")
