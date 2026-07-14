# Databricks notebook source
import json
import time
import requests

schema = "workspace.mlpabcef85c"
table_name = "profilesaa70e4"
full_table_name = f"{schema}.{table_name}"
prefix = "mlpabcef85c"
endpoint_name = f"{prefix}-profilesaa70e4-ep"

ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
token = ctx.apiToken().get()
host = ctx.apiUrl().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
print(f"Host: {host}")

# COMMAND ----------
# Step 1: Check if online table exists and its state
resp = requests.get(f"{host}/api/2.0/online-tables/{full_table_name}", headers=headers)
print(f"Online table GET: {resp.status_code} {resp.text[:500]}")

# COMMAND ----------
# Step 2: Create online table if needed
if resp.status_code != 200:
    create_payload = {
        "name": full_table_name,
        "spec": {
            "source_table_full_name": full_table_name,
            "primary_key_columns": ["account_id"],
            "run_triggered": {}
        }
    }
    cr = requests.post(f"{host}/api/2.0/online-tables", headers=headers, json=create_payload)
    print(f"Create online table: {cr.status_code}")
    print(cr.text[:2000])
else:
    ot = resp.json()
    state = ot.get("status", {}).get("detailed_state", "")
    print(f"Online table exists, state: {state}")

# COMMAND ----------
# Step 3: Poll until ONLINE
for i in range(30):
    time.sleep(10)
    poll = requests.get(f"{host}/api/2.0/online-tables/{full_table_name}", headers=headers)
    if poll.status_code != 200:
        print(f"Poll error: {poll.status_code} {poll.text[:200]}")
        break
    pd = poll.json()
    state = pd.get("status", {}).get("detailed_state", "UNKNOWN")
    print(f"[{(i+1)*10}s] state={state}")
    if state in ["ONLINE", "ONLINE_NO_PENDING_UPDATE", "ONLINE_CONTINUOUS_UPDATE"]:
        print("ONLINE!")
        print(json.dumps(pd.get("status", {}), indent=2))
        break
    if "FAIL" in str(state).upper():
        print(f"FAILED: {json.dumps(pd, indent=2)[:2000]}")
        break

# COMMAND ----------
# Step 4: Check the online table details for query endpoint info
final = requests.get(f"{host}/api/2.0/online-tables/{full_table_name}", headers=headers)
print(f"Final online table state: {final.status_code}")
print(json.dumps(final.json(), indent=2)[:3000])

# COMMAND ----------
# Step 5: Try querying rows from the online table
lookup_keys = ["A0003","A0005","A0012","A0015","A0023","A0030","A0031","A0034","A0048","A0049","A0055","A0063","A0066","A0072","A0085","A0090","A0103","A0109","A0112","A0113"]

# Test row lookup via GET with params
test_key = lookup_keys[0]
row_resp = requests.get(
    f"{host}/api/2.0/online-tables/{full_table_name}/rows",
    headers=headers,
    params={"account_id": test_key}
)
print(f"Row GET status: {row_resp.status_code}")
print(f"Row GET response: {row_resp.text[:500]}")

# COMMAND ----------
# Step 6: Create a Feature Spec for the feature serving endpoint
feature_spec_name = f"{schema}.{prefix}_feat_spec"
feature_spec_payload = {
    "name": feature_spec_name,
    "features": [
        {
            "name": "profile_lookup",
            "feature_lookup": {
                "table_name": full_table_name,
                "lookup_key": "account_id",
                "feature_names": ["f1", "f2", "f3", "f4"]
            }
        }
    ]
}

# Check if feature spec exists
fs_check = requests.get(
    f"{host}/api/2.0/feature-store/feature-specs/{feature_spec_name}",
    headers=headers
)
print(f"Feature spec check: {fs_check.status_code} {fs_check.text[:300]}")

if fs_check.status_code != 200:
    fs_create = requests.post(
        f"{host}/api/2.0/feature-store/feature-specs",
        headers=headers,
        json=feature_spec_payload
    )
    print(f"Feature spec create: {fs_create.status_code} {fs_create.text[:500]}")

# COMMAND ----------
# Step 7: Create a Feature Serving Endpoint
ep_check = requests.get(f"{host}/api/2.0/serving-endpoints/{endpoint_name}", headers=headers)
print(f"Endpoint check: {ep_check.status_code}")

if ep_check.status_code != 200:
    ep_payload = {
        "name": endpoint_name,
        "config": {
            "served_entities": [
                {
                    "name": "profile-lookup",
                    "feature_spec_name": feature_spec_name
                }
            ]
        }
    }
    ep_create = requests.post(f"{host}/api/2.0/serving-endpoints", headers=headers, json=ep_payload)
    print(f"Endpoint create: {ep_create.status_code} {ep_create.text[:1000]}")

# COMMAND ----------
# Step 8: Wait for the serving endpoint to be ready
for i in range(30):
    time.sleep(10)
    ep_poll = requests.get(f"{host}/api/2.0/serving-endpoints/{endpoint_name}", headers=headers)
    if ep_poll.status_code != 200:
        print(f"Endpoint poll error: {ep_poll.status_code} {ep_poll.text[:200]}")
        break
    ep_data = ep_poll.json()
    ep_state = ep_data.get("state", {}).get("ready", "NOT_READY")
    print(f"[{(i+1)*10}s] Endpoint state: {ep_state}")
    if ep_state == "READY":
        print("Endpoint is READY!")
        break

# COMMAND ----------
# Step 9: Query the feature serving endpoint for each key
results = {}
with open("/Workspace/Users/benedict@logicalclocks.com/mlpabcef85c/lookup_keys.txt") as f:
    lookup_keys = [line.strip() for line in f if line.strip()]

for key in lookup_keys:
    query_resp = requests.post(
        f"{host}/api/2.0/serving-endpoints/{endpoint_name}/invocations",
        headers=headers,
        json={"dataframe_records": [{"account_id": key}]}
    )
    if query_resp.status_code == 200:
        resp_data = query_resp.json()
        if "predictions" in resp_data:
            pred = resp_data["predictions"]
            if isinstance(pred, list) and pred:
                p = pred[0]
                results[key] = [p.get("f1"), p.get("f2"), p.get("f3"), p.get("f4")]
        else:
            print(f"Key {key} unexpected response: {json.dumps(resp_data)[:300]}")
    else:
        print(f"Key {key} error: {query_resp.status_code} {query_resp.text[:200]}")

print(f"Got {len(results)} results")
print(json.dumps(results, indent=2)[:2000])

# COMMAND ----------
# Step 10: Save results
output = {"vectors": results}
with open("/Workspace/Users/benedict@logicalclocks.com/mlpabcef85c/answers.json", "w") as f:
    json.dump(output, f)
print("Saved answers.json")
dbutils.notebook.exit(json.dumps(output)[:500])
