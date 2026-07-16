import json
import requests
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
host = w.config.host
token = w.config.token

headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

results = []

synced_payload = {
    "name": "workspace.mlpabbb38f1.derivedd05474_online",
    "spec": {
        "source_table_full_name": "workspace.mlpabbb38f1.derivedd05474",
        "primary_key_columns": ["row_id"],
        "run_triggered": {}
    }
}

paths = [
    "/api/2.0/synced-tables",
    "/api/2.1/synced-tables",
    "/api/2.0/preview/synced-tables",
    "/api/2.0/preview/unity-catalog/synced-tables",
]

for path in paths:
    url = f"{host}{path}"
    r = requests.post(url, json=synced_payload, headers=headers)
    results.append(f"POST {path}: {r.status_code} -> {r.text[:300]}")
    if r.status_code in (200, 201):
        break

dbutils.notebook.exit(json.dumps(results))
