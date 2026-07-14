import os
import json
import requests
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
host = w.config.host
token = w.config.token

headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

# Try synced tables API
synced_payload = {
    "name": "workspace.mlpabbb38f1.derivedd05474_online",
    "spec": {
        "source_table_full_name": "workspace.mlpabbb38f1.derivedd05474",
        "primary_key_columns": ["row_id"],
        "run_triggered": {}
    }
}

# Try multiple API paths
paths = [
    "/api/2.0/synced-tables",
    "/api/2.1/synced-tables",
    "/api/2.0/preview/synced-tables",
    "/api/2.0/preview/unity-catalog/synced-tables",
]

for path in paths:
    url = f"{host}{path}"
    r = requests.post(url, json=synced_payload, headers=headers)
    print(f"POST {path}: {r.status_code} -> {r.text[:200]}")

# Also try online tables (even though deprecated)
online_payload = {
    "table": {
        "name": "workspace.mlpabbb38f1.derivedd05474_online",
        "spec": {
            "source_table_full_name": "workspace.mlpabbb38f1.derivedd05474",
            "primary_key_columns": ["row_id"],
            "run_triggered": {}
        }
    }
}
r = requests.post(f"{host}/api/2.0/online-tables", json=online_payload, headers=headers)
print(f"POST /api/2.0/online-tables: {r.status_code} -> {r.text[:200]}")
