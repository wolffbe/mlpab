# Databricks notebook source
# COMMAND ----------
import json, time
import requests

host = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
base_url = f"https://{host}"
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

schema = "workspace.mlpabae7d2f"
table_name = f"{schema}.profilesaa70e4"
online_store_name = "mlpabae7d2f-store"
online_table_name_uc = f"{schema}.profilesaa70e4_online"
output_path = "/Volumes/workspace/mlpabae7d2f/mlpabae7d2f_vol/answers.json"

lookup_keys = [
    "A0003", "A0005", "A0012", "A0015", "A0023", "A0030", "A0031", "A0034",
    "A0048", "A0049", "A0055", "A0063", "A0066", "A0072", "A0085", "A0090",
    "A0103", "A0109", "A0112", "A0113"
]

log = []

# COMMAND ----------
# Step 1: Register feature table with primary keys via REST API
body = {
    "name": table_name,
    "primary_keys": [{"name": "account_id", "data_type": "STRING"}],
    "description": "Account feature profiles v1"
}
r = requests.post(f"{base_url}/api/2.0/feature-store/feature-tables", headers=headers, json=body)
log.append(f"Create feature-table -> {r.status_code}: {r.text[:500]}")

# Also try without namespace prefix to see if workspace feature store needs it
if r.status_code not in (200, 201):
    # Try the tables endpoint
    r2 = requests.post(f"{base_url}/api/2.0/feature-store/tables", headers=headers, json={"table": body})
    log.append(f"Create table -> {r2.status_code}: {r2.text[:500]}")

# COMMAND ----------
# Get table state
r3 = requests.get(f"{base_url}/api/2.0/feature-store/feature-tables/{table_name}", headers=headers)
log.append(f"Get feature-table -> {r3.status_code}: {r3.text[:1000]}")

r4 = requests.get(f"{base_url}/api/2.0/feature-store/tables/{table_name}", headers=headers)
log.append(f"Get tables -> {r4.status_code}: {r4.text[:1000]}")

# COMMAND ----------
# Step 2: Publish to online store
publish_body = {
    "publish_spec": {
        "online_store": online_store_name,
        "online_table_name": online_table_name_uc,
        "publish_mode": "OVERWRITE"
    }
}
r5 = requests.post(f"{base_url}/api/2.0/feature-store/tables/{table_name}/publish", headers=headers, json=publish_body)
log.append(f"Publish -> {r5.status_code}: {r5.text[:2000]}")

# COMMAND ----------
# Print all logs so far
for l in log:
    print(l)

# COMMAND ----------
# Step 3: Regardless of online store, get data from Delta for verification purposes
# But we need to go through online path - let's check what's available
print("\n--- Checking online store ---")

r6 = requests.get(f"{base_url}/api/2.0/feature-store/online-stores/{online_store_name}", headers=headers)
print(f"Online store state: {r6.status_code}: {r6.text[:1000]}")

# Check if there's a lookup endpoint on the online store
r7 = requests.get(f"{base_url}/api/2.0/feature-store/online-stores/{online_store_name}/feature-tables", headers=headers)
print(f"Online store feature-tables: {r7.status_code}: {r7.text[:2000]}")

# COMMAND ----------
# Try the online lookup endpoint
print("\n--- Testing online lookup endpoints ---")
test_key = "A0003"

# Different potential lookup paths
paths_to_try = [
    f"/api/2.0/feature-store/online-stores/{online_store_name}/feature-tables/{table_name}/lookup",
    f"/api/2.0/feature-store/tables/{table_name}/online-features",
    f"/api/2.0/feature-store/feature-tables/{table_name}/online-lookup",
    f"/api/2.0/feature-store/online-stores/{online_store_name}/lookup",
]

for path in paths_to_try:
    try:
        r = requests.get(f"{base_url}{path}", headers=headers, params={"account_id": test_key})
        print(f"GET {path} -> {r.status_code}: {r.text[:300]}")
        r = requests.post(f"{base_url}{path}", headers=headers, json={"lookup_key": {"account_id": test_key}})
        print(f"POST {path} -> {r.status_code}: {r.text[:300]}")
    except Exception as e:
        print(f"Error: {path}: {e}")

dbutils.notebook.exit(json.dumps({"log": log[:10]}))
