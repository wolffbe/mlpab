"""Probe online table endpoints and try synced tables."""
import os
import requests

host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
if not host.startswith("http"):
    host = "https://" + host
token = os.environ.get("DATABRICKS_TOKEN", "")
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]
SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]
CATALOG = SCHEMA.split(".")[0]
DB = SCHEMA.split(".")[1]
PRED_NAME = "airqpredf4aae3"
source_table = f"{CATALOG}.{DB}.{PRED_NAME}"

print(f"Host: {host[:50]}")
print(f"Source: {source_table}")
print()

# Correct GET endpoint from SDK source: /api/2.0/online-tables/{name}
r = requests.get(f"{host}/api/2.0/online-tables/{source_table}", headers=headers)
print(f"GET /api/2.0/online-tables/{source_table}: {r.status_code}")
print(f"  {r.text[:300]}")
print()

# Try POST with full path (correct per SDK source)
payload = {
    "name": source_table,
    "spec": {
        "source_table_full_name": source_table,
        "primary_key_columns": ["date"],
        "run_triggered": {}
    }
}
r2 = requests.post(f"{host}/api/2.0/online-tables", headers=headers, json=payload)
print(f"POST /api/2.0/online-tables: {r2.status_code}")
print(f"  {r2.text[:400]}")
print()

# Check w.database API
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
db_methods = [m for m in dir(w.database) if not m.startswith("_")]
print(f"w.database methods: {db_methods}")
print()

# Look at the "database" service - it might be related to Lakebase/online tables
import inspect
db_src = inspect.getsource(type(w.database))
print(f"database service (first 2000):")
print(db_src[:2000])
