# Databricks notebook source
# COMMAND ----------
import requests
import json

host = spark.conf.get("spark.databricks.workspaceUrl", "")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

results = {}
table_name = "workspace.mlpabb40f43.recs708df6"

# COMMAND ----------
# Try feature specs API
feature_spec_paths = [
    "/api/2.0/feature-store/feature-specs",
    "/api/2.0/serving-endpoints/feature-serving",
    "/api/2.0/feature-specs",
    "/api/2.1/feature-specs",
]

for path in feature_spec_paths:
    resp = requests.get(f"https://{host}{path}", headers=headers)
    results[f"GET {path}"] = f"{resp.status_code}: {resp.text[:200]}"
    print(f"GET {path}: {resp.status_code}")

# COMMAND ----------
# Try to create a Feature Spec
feature_spec_payload = {
    "name": f"{table_name}_spec",
    "features": [
        {
            "table_name": table_name,
            "feature_columns": ["rec_id", "user_id", "rank", "item_id"],
            "lookup_key": ["rec_id"]
        }
    ]
}

for post_path in ["/api/2.0/feature-specs", "/api/2.1/feature-specs"]:
    resp = requests.post(f"https://{host}{post_path}", headers=headers, json=feature_spec_payload)
    results[f"POST {post_path}"] = f"{resp.status_code}: {resp.text[:300]}"
    print(f"POST {post_path}: {resp.status_code}: {resp.text[:300]}")

# COMMAND ----------
# Look for synced table API via different discovery method
# Try GET on specific table name route for online tables
table_encoded = table_name.replace(".", "%2E")
resp = requests.get(f"https://{host}/api/2.0/online-tables/tables/{table_name}", headers=headers)
results["GET online-tables by name"] = f"{resp.status_code}: {resp.text[:300]}"
print(f"GET online table by name: {resp.status_code}: {resp.text[:300]}")

# COMMAND ----------
# Check if we can find synced table API paths through the error structure
# Some APIs might have helpful "see also" in their error responses
resp = requests.post(f"https://{host}/api/2.0/online-tables/tables", headers=headers, json={
    "name": "workspace.mlpabb40f43.recs708df6_online",
    "spec": {
        "source_table_full_name": table_name,
        "primary_key_columns": ["rec_id"],
        "run_triggered": {}
    }
})
results["POST online-tables v2"] = f"{resp.status_code}: {resp.text[:400]}"
print(f"POST online tables: {resp.status_code}: {resp.text[:400]}")

# COMMAND ----------
# Try creating a serving endpoint for feature lookup
endpoint_payload = {
    "name": "mlpabb40f43_recs708df6_serving",
    "config": {
        "served_entities": [
            {
                "feature_spec_name": f"{table_name}_spec",
                "workload_size": "Small",
                "scale_to_zero_enabled": True
            }
        ]
    }
}

resp = requests.post(f"https://{host}/api/2.0/serving-endpoints", headers=headers, json=endpoint_payload)
results["POST serving-endpoints with feature_spec"] = f"{resp.status_code}: {resp.text[:400]}"
print(f"POST serving-endpoints with feature_spec: {resp.status_code}: {resp.text[:400]}")

# COMMAND ----------
# Try to list synced tables by checking the catalog API
resp = requests.get(f"https://{host}/api/2.1/online-tables", headers=headers)
results["GET /api/2.1/online-tables direct"] = f"{resp.status_code}: {resp.text[:300]}"
print(f"GET /api/2.1/online-tables: {resp.status_code}: {resp.text[:300]}")

# COMMAND ----------
# Try the actual online-tables CLI endpoint path
# The CLI for "databricks online-tables create" internally calls an API
# Let's check what path the CLI is using by looking at the version
resp = requests.get(f"https://{host}/api/2.0/online-tables", headers=headers)
results["GET /api/2.0/online-tables direct 2"] = f"{resp.status_code}: {resp.text[:300]}"
print(f"GET /api/2.0/online-tables: {resp.status_code}: {resp.text[:300]}")

dbutils.notebook.exit(json.dumps(results, indent=2))
