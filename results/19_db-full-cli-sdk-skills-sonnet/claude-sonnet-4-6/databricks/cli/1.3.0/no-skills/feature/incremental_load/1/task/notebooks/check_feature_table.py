# Databricks notebook source
# Check feature table registration and explore serving options

import requests
import json

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

catalog_name = "workspace"
schema_name = "mlpab312fe6"
table_name = "incremental3526e9"
full_name = f"{catalog_name}.{schema_name}.{table_name}"
prefix = "mlpab312fe6"

results = {}

# Check the feature table properties in UC
resp = requests.get(
    f"{host}/api/2.1/unity-catalog/tables/{full_name}",
    headers=headers, timeout=30
)
results['table_info'] = {"status": resp.status_code, "body": resp.json() if resp.ok else resp.text}

# Check if feature store metadata exists
resp2 = requests.get(
    f"{host}/api/2.0/feature-store/feature-tables/{full_name}",
    headers=headers, timeout=30
)
results['fs_metadata'] = {"status": resp2.status_code, "body": resp2.text[:500]}

# Try to create feature serving endpoint (for real-time lookup)
# Feature Serving endpoints serve Unity Catalog feature tables
endpoint_name = f"{prefix}_feat_svc"
serving_payload = {
    "name": endpoint_name,
    "config": {
        "served_entities": [
            {
                "entity_name": full_name,
                "workload_type": "CPU",
                "workload_size": "Small",
                "scale_to_zero_enabled": True
            }
        ]
    }
}

resp3 = requests.post(
    f"{host}/api/2.0/serving-endpoints",
    headers=headers, json=serving_payload, timeout=30
)
results['feature_serving_endpoint'] = {"status": resp3.status_code, "body": resp3.text[:1000]}

# Try with feature_spec instead
serving_payload2 = {
    "name": f"{endpoint_name}_v2",
    "config": {
        "served_entities": [
            {
                "feature_spec": {
                    "feature_group": full_name
                },
                "workload_size": "Small",
                "scale_to_zero_enabled": True
            }
        ]
    }
}

resp4 = requests.post(
    f"{host}/api/2.0/serving-endpoints",
    headers=headers, json=serving_payload2, timeout=30
)
results['feature_serving_v2'] = {"status": resp4.status_code, "body": resp4.text[:1000]}

dbutils.notebook.exit(json.dumps(results, indent=2))
