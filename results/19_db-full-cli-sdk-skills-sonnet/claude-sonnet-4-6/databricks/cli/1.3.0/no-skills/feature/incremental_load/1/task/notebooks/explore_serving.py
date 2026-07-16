# Databricks notebook source

import requests
import json

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

catalog_name = "workspace"
schema_name = "mlpab312fe6"
source_table = f"{catalog_name}.{schema_name}.incremental3526e9"

results = {}

# Try feature serving endpoint
serving_payload = {
    "name": f"mlpab312fe6_incremental3526e9_serving",
    "config": {
        "served_entities": [
            {
                "entity_name": source_table,
                "entity_version": "1",
                "scale_to_zero_enabled": True,
                "workload_size": "Small"
            }
        ]
    }
}

# Try creating feature serving endpoint
resp = requests.post(f"{host}/api/2.0/serving-endpoints", headers=headers, json=serving_payload, timeout=30)
results['feature_serving'] = {"status": resp.status_code, "body": resp.text[:1000]}

# Try synced tables paths
sync_paths = [
    ("/api/2.0/serving-endpoints", "GET"),
    ("/api/2.0/catalog-enabled-online-stores", "GET"),
    ("/api/2.0/catalog/tables", "GET"),
    ("/api/2.1/unity-catalog/synced-tables", "GET"),
    ("/api/2.0/sql/synced-tables", "GET"),
    ("/api/2.0/unity-catalog/synced-tables", "GET"),
]

for path, method in sync_paths:
    resp = requests.request(method, f"{host}{path}", headers=headers, timeout=10)
    results[f"{method}_{path}"] = {"status": resp.status_code, "preview": resp.text[:200]}

dbutils.notebook.exit(json.dumps(results, indent=2))
