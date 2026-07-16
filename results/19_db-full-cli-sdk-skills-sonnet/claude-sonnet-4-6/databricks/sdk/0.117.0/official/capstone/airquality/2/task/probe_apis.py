"""Probe REST API endpoints for Synced Tables and alternatives."""
import os
import requests

host_raw = os.environ.get("DATABRICKS_HOST", "")
# Ensure host has scheme
if not host_raw.startswith("http"):
    host = f"https://{host_raw}"
else:
    host = host_raw.rstrip("/")

token = os.environ.get("DATABRICKS_TOKEN", "")
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]
SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]
CATALOG = SCHEMA.split(".")[0]
DB = SCHEMA.split(".")[1]
PRED_NAME = "airqpredf4aae3"
source_table = f"{CATALOG}.{DB}.{PRED_NAME}"

print(f"Host: {host[:40]}")
print(f"Source: {source_table}")
print()

# Try various REST paths for Synced Tables
paths_to_try = [
    ("GET", f"/api/2.0/catalog/synced-tables"),
    ("GET", f"/api/2.0/synced-tables/tables"),
    ("GET", f"/api/2.0/online-tables/tables/{source_table}"),
    ("GET", f"/api/2.0/online-tables/tables"),
    ("GET", f"/api/2.1/online-tables/tables"),
    ("GET", f"/api/2.0/feature-store/tables/{source_table}"),
    ("GET", f"/api/2.0/feature-store/online-tables"),
    ("GET", f"/api/2.0/feature-store/online-stores/{PREFIX}-pred-store"),
    ("GET", f"/api/2.0/serving-endpoints"),
]

for method, path in paths_to_try:
    try:
        r = requests.request(
            method, f"{host}{path}",
            headers=headers,
            params={"catalog_name": CATALOG, "schema_name": DB} if "tables" in path and "?" not in path else {},
            timeout=15
        )
        print(f"{method} {path}: {r.status_code} {r.text[:200]}")
    except Exception as e:
        print(f"{method} {path}: ERROR {e}")
    print()

# Try POST for Synced Tables
print("--- POST attempts ---")
for post_path in [
    "/api/2.0/catalog/synced-tables",
    "/api/2.0/synced-tables/tables",
]:
    payload = {
        "name": source_table,
        "spec": {
            "source_table_full_name": source_table,
            "primary_key_columns": ["date"],
        }
    }
    try:
        r = requests.post(f"{host}{post_path}", headers=headers, json=payload, timeout=15)
        print(f"POST {post_path}: {r.status_code} {r.text[:300]}")
    except Exception as e:
        print(f"POST {post_path}: ERROR {e}")
    print()
