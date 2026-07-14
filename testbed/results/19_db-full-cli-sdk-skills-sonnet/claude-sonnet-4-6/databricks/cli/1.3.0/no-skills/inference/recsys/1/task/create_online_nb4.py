# Databricks notebook source
# COMMAND ----------
# Try the feature_store module (older API) and find online access solution
import requests
import json

host = spark.conf.get("spark.databricks.workspaceUrl", "")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

results = {}

# COMMAND ----------
# Check if feature_store module is available
try:
    from databricks import feature_store
    fs = feature_store.FeatureStoreClient()
    methods = [m for m in dir(fs) if not m.startswith('_')]
    results["feature_store_methods"] = methods
    print(f"Feature Store client available with methods: {methods}")
except Exception as e:
    results["feature_store"] = f"Not available: {e}"
    print(f"Feature Store error: {e}")

# COMMAND ----------
# Try online tables via feature store REST API
for api_path in [
    "/api/2.0/feature-store/tables",
    "/api/2.0/feature-store/online-tables",
    "/api/2.0/ml/feature-store/online-stores",
]:
    resp = requests.get(f"https://{host}{api_path}", headers=headers)
    results[f"GET {api_path}"] = f"{resp.status_code}: {resp.text[:200]}"
    print(f"GET {api_path}: {resp.status_code}")

# COMMAND ----------
# Try creating online table via feature store REST API
table_name = "workspace.mlpabb40f43.recs708df6"
online_payload = {
    "table_name": table_name,
    "store_type": "DYNAMODB",
    "online_table_specs": {
        "table_name": table_name,
        "primary_keys": ["rec_id"]
    }
}

for post_path in [
    "/api/2.0/feature-store/online-tables",
    "/api/2.0/ml/feature-store/online-stores",
]:
    resp = requests.post(f"https://{host}{post_path}", headers=headers, json=online_payload)
    results[f"POST {post_path}"] = f"{resp.status_code}: {resp.text[:300]}"
    print(f"POST {post_path}: {resp.status_code}")

# COMMAND ----------
# Check if there's a Databricks-specific online table API
# by looking at the online-tables CLI path
# The CLI uses /api/2.0/online-tables/tables/{name}

resp = requests.get(f"https://{host}/api/2.0/online-tables/tables", headers=headers)
results["GET /api/2.0/online-tables/tables"] = f"{resp.status_code}: {resp.text[:200]}"
print(f"GET /api/2.0/online-tables/tables: {resp.status_code}: {resp.text[:200]}")

# Test the actual online tables endpoint that the CLI uses
resp = requests.get(f"https://{host}/api/2.0/online-tables/tables/{table_name}", headers=headers)
results[f"GET /api/2.0/online-tables/tables/{table_name}"] = f"{resp.status_code}: {resp.text[:300]}"
print(f"GET /api/2.0/online-tables/tables/{table_name}: {resp.status_code}")

# Try POST for online table create
online_payload2 = {
    "name": "workspace.mlpabb40f43.recs708df6_online",
    "spec": {
        "source_table_full_name": table_name,
        "primary_key_columns": ["rec_id"],
        "run_triggered": {}
    }
}

resp = requests.post(f"https://{host}/api/2.0/online-tables/tables", headers=headers, json=online_payload2)
results["POST /api/2.0/online-tables/tables"] = f"{resp.status_code}: {resp.text[:400]}"
print(f"POST /api/2.0/online-tables/tables: {resp.status_code}: {resp.text[:400]}")

dbutils.notebook.exit(json.dumps(results, indent=2))
