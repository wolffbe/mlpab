import json
import requests
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
host = w.config.host
token = w.config.token
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

results = {}

# Check what online_tables methods exist
results["sdk_online_tables"] = dir(w.online_tables)

# Check what synced_tables methods exist
try:
    results["sdk_synced_tables"] = dir(w.synced_tables)
except AttributeError:
    results["sdk_synced_tables"] = "not available"

# Try GET on various paths to see what's available
test_paths = [
    "/api/2.0/online-tables",
    "/api/2.0/synced-tables",
    "/api/2.0/preview/online-tables",
    "/api/2.1/online-tables",
]
for path in test_paths:
    r = requests.get(f"{host}{path}", headers=headers)
    results[f"GET {path}"] = f"{r.status_code}: {r.text[:100]}"

dbutils.notebook.exit(json.dumps(results, indent=2))
