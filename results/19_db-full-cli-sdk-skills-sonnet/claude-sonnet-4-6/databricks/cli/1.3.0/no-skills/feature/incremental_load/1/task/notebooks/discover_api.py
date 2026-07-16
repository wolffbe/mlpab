# Databricks notebook source

import requests
import json

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()

headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

catalog = "workspace"
schema = "mlpab312fe6"
source_table = f"{catalog}.{schema}.incremental3526e9"
online_name = f"{catalog}.{schema}.incremental3526e9_online"

results = {}

# Try SDK
try:
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service import catalog as sdk_catalog
    w = WorkspaceClient()

    # Check what services are available
    services = [attr for attr in dir(w) if not attr.startswith('_')]
    results['sdk_services'] = [s for s in services if 'online' in s.lower() or 'sync' in s.lower() or 'feature' in s.lower()]

    # Try online_tables
    try:
        r = w.online_tables.create(
            name=online_name,
            spec=sdk_catalog.OnlineTableSpec(
                source_table_full_name=source_table,
                primary_key_columns=["row_id"],
                run_triggered=sdk_catalog.OnlineTableSpecTriggeredSchedulingPolicy()
            )
        )
        results['sdk_online_table'] = str(r)
    except Exception as e:
        results['sdk_online_table_error'] = str(e)

except Exception as e:
    results['sdk_error'] = str(e)

# Try REST API endpoints
test_endpoints = [
    ("POST", "/api/2.0/online-tables", {"name": online_name, "spec": {"source_table_full_name": source_table, "primary_key_columns": ["row_id"], "run_triggered": {}}}),
    ("GET", "/api/2.0/catalog/online-tables", None),
    ("GET", "/api/2.1/unity-catalog/online-tables", None),
    ("POST", "/api/2.0/sql/synced-tables", {"name": online_name, "source_table": source_table}),
]

for method, ep, payload in test_endpoints:
    url = f"{host}{ep}"
    try:
        if method == "POST":
            resp = requests.post(url, headers=headers, json=payload, timeout=15)
        else:
            resp = requests.get(url, headers=headers, timeout=15)
        results[f"{method}_{ep}"] = {"status": resp.status_code, "body": resp.text[:500]}
    except Exception as e:
        results[f"{method}_{ep}"] = {"error": str(e)}

dbutils.notebook.exit(json.dumps(results, indent=2))
