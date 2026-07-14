# Databricks notebook source
# COMMAND ----------
import json, os, time
import requests

# Get workspace URL and token from environment
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

print(f"Host: {base_url}")

# COMMAND ----------
# Step 1: Register feature table with primary key in Feature Store
# Try creating feature table metadata via REST API

def api_post(path, body):
    r = requests.post(f"{base_url}{path}", headers=headers, json=body)
    print(f"POST {path} -> {r.status_code}: {r.text[:500]}")
    return r

def api_get(path):
    r = requests.get(f"{base_url}{path}", headers=headers)
    print(f"GET {path} -> {r.status_code}: {r.text[:500]}")
    return r

# Try to get current state
r = api_get(f"/api/2.0/feature-store/tables/{table_name}")
print(f"\nCurrent table state:")
if r.status_code == 200:
    print(r.json())

# COMMAND ----------
# Create/update feature table with primary key
body = {
    "table": {
        "name": table_name,
        "primary_keys": [{"name": "account_id", "data_type": "STRING"}],
        "description": "Account feature profiles v1"
    }
}

# Try PUT to update
r = requests.put(f"{base_url}/api/2.0/feature-store/tables/{table_name}", headers=headers, json=body)
print(f"PUT -> {r.status_code}: {r.text[:1000]}")

if r.status_code not in (200, 201):
    # Try POST to create
    r2 = requests.post(f"{base_url}/api/2.0/feature-store/tables", headers=headers, json=body)
    print(f"POST /tables -> {r2.status_code}: {r2.text[:1000]}")

# COMMAND ----------
# Step 2: Publish to online store
publish_body = {
    "publish_spec": {
        "online_store": online_store_name,
        "online_table_name": online_table_name_uc,
        "publish_mode": "OVERWRITE"
    }
}
r = requests.post(f"{base_url}/api/2.0/feature-store/tables/{table_name}/publish", headers=headers, json=publish_body)
print(f"Publish -> {r.status_code}: {r.text[:2000]}")

if r.status_code not in (200, 201):
    # Try SNAPSHOT mode
    publish_body["publish_spec"]["publish_mode"] = "SNAPSHOT"
    r = requests.post(f"{base_url}/api/2.0/feature-store/tables/{table_name}/publish", headers=headers, json=publish_body)
    print(f"Publish SNAPSHOT -> {r.status_code}: {r.text[:2000]}")

# COMMAND ----------
# Wait for publish to complete
import time
for i in range(60):
    r = api_get(f"/api/2.0/feature-store/tables/{table_name}/online-table/{online_table_name_uc}")
    if r.status_code == 200:
        state = r.json().get("state", "")
        print(f"Attempt {i}: state={state}")
        if state in ("ONLINE", "PUBLISHED", "ACTIVE"):
            print("Online table is ready!")
            break
    time.sleep(10)

# COMMAND ----------
# Step 3: Query the online store
results = {}
for account_id in lookup_keys:
    query_body = {
        "table_name": table_name,
        "lookup_key": {"account_id": account_id},
        "online_store_name": online_store_name
    }
    r = requests.post(f"{base_url}/api/2.0/feature-store/tables/{table_name}/online-features", headers=headers, json=query_body)
    print(f"Query {account_id} -> {r.status_code}: {r.text[:500]}")
    if r.status_code == 200:
        features = r.json()
        results[account_id] = [float(features.get("f1", 0)), float(features.get("f2", 0)),
                               float(features.get("f3", 0)), float(features.get("f4", 0))]

print(f"\nResults collected: {len(results)} entries")

# COMMAND ----------
# Write results to volume
output = {"vectors": results}
with open(output_path, "w") as fh:
    json.dump(output, fh)
print(f"Results written to {output_path}")
print(json.dumps(output, indent=2))
