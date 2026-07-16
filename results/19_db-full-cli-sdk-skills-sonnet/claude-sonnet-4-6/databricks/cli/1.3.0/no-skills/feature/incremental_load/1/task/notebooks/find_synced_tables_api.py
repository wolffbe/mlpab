# Databricks notebook source
# Discover the Synced Tables API

import requests
import json

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

catalog_name = "workspace"
schema_name = "mlpab312fe6"
full_name = f"{catalog_name}.{schema_name}.incremental3526e9"
online_name = f"{catalog_name}.{schema_name}.incremental3526e9_online"

results = {}

# Check SDK for synced tables
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

# List all available services
sdk_attrs = [attr for attr in dir(w) if not attr.startswith('_')]
results['sdk_all_services'] = sdk_attrs

# Check for synced_tables
if hasattr(w, 'synced_tables'):
    results['has_synced_tables'] = True
    try:
        r = w.synced_tables.create(name=online_name, spec={
            "source_table_full_name": full_name,
            "primary_key_columns": ["row_id"]
        })
        results['synced_table_created'] = str(r)
    except Exception as e:
        results['synced_tables_error'] = str(e)
else:
    results['has_synced_tables'] = False

# Try API endpoints I haven't tried yet
more_paths = [
    "/api/2.0/catalog/synced-tables",
    "/api/2.0/tables/online",
    "/api/2.0/serving-endpoints/feature",
    "/api/2.0/ml/feature-store/tables",
    "/api/2.0/preview/online-tables",
    "/api/2.0/feature-engineering/feature-tables",
    "/api/3.0/online-tables",
    "/api/2.0/databricks-feature-engineering",
]

for path in more_paths:
    try:
        resp = requests.get(f"{host}{path}", headers=headers, timeout=5)
        results[f"GET_{path}"] = resp.status_code
        if resp.status_code == 200:
            results[f"GET_{path}_body"] = resp.text[:300]
    except Exception as e:
        results[f"GET_{path}"] = str(e)

# Try POST to synced-tables variants
synced_payload = {
    "name": online_name,
    "spec": {
        "source_table_full_name": full_name,
        "primary_key_columns": ["row_id"],
        "run_triggered": {}
    }
}

for path in ["/api/2.0/catalog/synced-tables", "/api/2.0/synced-tables", "/api/2.1/catalog/synced-tables"]:
    try:
        resp = requests.post(f"{host}{path}", headers=headers, json=synced_payload, timeout=5)
        results[f"POST_{path}"] = {"status": resp.status_code, "body": resp.text[:200]}
    except Exception as e:
        results[f"POST_{path}"] = str(e)

dbutils.notebook.exit(json.dumps(results, indent=2)[:5000])
