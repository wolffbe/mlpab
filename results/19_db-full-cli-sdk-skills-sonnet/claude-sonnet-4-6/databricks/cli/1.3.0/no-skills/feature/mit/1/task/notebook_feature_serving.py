# Databricks notebook source
# COMMAND ----------
import json, requests

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()

results = {}

# Test feature serving API paths
feature_serving_paths = [
    "/api/2.0/feature-serving-endpoints",
    "/api/2.0/serving-endpoints",
]

# Try creating a Feature Serving Endpoint for online lookup
payload = {
    "name": "mlpabf1452c_featuresb1ea93",
    "config": {
        "served_entities": [{
            "name": "featuresb1ea93",
            "feature_spec": {
                "feature_view": "workspace.mlpabf1452c.featuresb1ea93",
                "entities": [{"name": "row_id", "type": "string"}],
                "feature_columns": ["amount_usd", "is_weekend", "amount_7d"],
                "lookup_key": ["row_id"]
            }
        }]
    }
}

for path in feature_serving_paths:
    try:
        r = requests.post(
            f"{host}{path}",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        results[path] = {"status": r.status_code, "body": r.text[:500]}
    except Exception as e:
        results[path] = {"error": str(e)}

# Also test GET for existing feature serving endpoints
try:
    r = requests.get(
        f"{host}/api/2.0/serving-endpoints",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30
    )
    results['list_serving'] = {"status": r.status_code, "body": r.text[:300]}
except Exception as e:
    results['list_serving'] = {"error": str(e)}

dbutils.notebook.exit(json.dumps(results))
