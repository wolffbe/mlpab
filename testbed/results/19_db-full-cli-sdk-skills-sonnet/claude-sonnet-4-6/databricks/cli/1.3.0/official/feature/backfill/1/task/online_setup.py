# Databricks notebook source
# Create online access for accountse81ff1 feature table

# COMMAND ----------
# Try Feature Engineering client for online table creation
try:
    from databricks.feature_engineering import FeatureEngineeringClient
    from databricks.feature_engineering.entities.feature_serving_endpoint import ServedEntity
    fe = FeatureEngineeringClient()
    print("Feature Engineering client available")

    # Check if table is registered
    ft = fe.get_table("workspace.mlpab0442b8.accountse81ff1")
    print(f"Feature table: {ft}")
except Exception as e:
    print(f"Feature Engineering SDK error: {e}")

# COMMAND ----------
# Try creating synced table via REST
import requests
import json

host = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

# Try different API paths for synced tables
api_paths = [
    "/api/2.0/synced-tables",
    "/api/2.1/synced-tables",
    "/api/2.0/uc/synced-tables",
    "/api/2.0/online-tables",
]

for path in api_paths:
    try:
        r = requests.get(f"https://{host}{path}", headers=headers)
        print(f"GET {path}: {r.status_code} - {r.text[:200]}")
    except Exception as e:
        print(f"GET {path}: ERROR {e}")

# COMMAND ----------
# Create synced table
payload = {
    "name": "workspace.mlpab0442b8.accountse81ff1_online",
    "spec": {
        "source_table_full_name": "workspace.mlpab0442b8.accountse81ff1",
        "primary_key_columns": ["row_id", "updated_at"],
        "run_triggered": {}
    }
}

for path in ["/api/2.0/synced-tables", "/api/2.1/synced-tables"]:
    r = requests.post(f"https://{host}{path}", headers=headers, json=payload)
    print(f"POST {path}: {r.status_code} - {r.text[:500]}")
