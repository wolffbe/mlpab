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

print(f"Host: {base_url}")

# COMMAND ----------
# Step 1: Register feature table with primary keys via REST API
# POST /api/2.0/feature-store/feature-tables (workspace feature store API)
body = {
    "name": table_name,
    "primary_keys": [{"name": "account_id", "data_type": "STRING"}],
    "description": "Account feature profiles v1"
}
r = requests.post(f"{base_url}/api/2.0/feature-store/feature-tables", headers=headers, json=body)
print(f"Create feature table -> {r.status_code}: {r.text[:2000]}")

# COMMAND ----------
# Verify registration
r2 = requests.get(f"{base_url}/api/2.0/feature-store/feature-tables/{table_name}", headers=headers)
print(f"Get feature table -> {r2.status_code}: {r2.text[:2000]}")

# COMMAND ----------
# Also try the tables endpoint (might be a different namespace)
r3 = requests.get(f"{base_url}/api/2.0/feature-store/tables/{table_name}", headers=headers)
print(f"Get table -> {r3.status_code}: {r3.text[:2000]}")

# COMMAND ----------
# Step 2: Publish to online store
for mode in ["PUBLISH_MODE_SNAPSHOT", "SNAPSHOT", "overwrite"]:
    publish_body = {
        "publish_spec": {
            "online_store": online_store_name,
            "online_table_name": online_table_name_uc,
            "publish_mode": mode
        }
    }
    r = requests.post(f"{base_url}/api/2.0/feature-store/tables/{table_name}/publish", headers=headers, json=publish_body)
    print(f"Publish mode={mode} -> {r.status_code}: {r.text[:2000]}")
    if r.status_code in (200, 201, 202):
        print("Publish succeeded!")
        break

# COMMAND ----------
# Wait for sync
time.sleep(30)

# Step 3: Query the online store for each lookup key
results = {}

# Try different query API paths
for account_id in lookup_keys:
    # Method 1: Direct lookup via feature-store API
    query_paths = [
        f"/api/2.0/feature-store/online-stores/{online_store_name}/feature-tables/{table_name}/lookup",
        f"/api/2.0/feature-store/tables/{table_name}/online-lookup",
        f"/api/2.0/feature-store/feature-tables/{table_name}/online-lookup",
    ]

    success = False
    for path in query_paths:
        # Try GET with query param
        r = requests.get(f"{base_url}{path}?account_id={account_id}", headers=headers)
        if r.status_code == 200:
            data = r.json()
            print(f"GET {path} -> {r.status_code}: {data}")
            if "f1" in data or "features" in data:
                feat = data.get("features", data)
                results[account_id] = [float(feat["f1"]), float(feat["f2"]), float(feat["f3"]), float(feat["f4"])]
                success = True
                break

        # Try POST with body
        r = requests.post(f"{base_url}{path}", headers=headers, json={"account_id": account_id})
        if r.status_code == 200:
            data = r.json()
            print(f"POST {path} -> {r.status_code}: {data}")
            if "f1" in data or "features" in data:
                feat = data.get("features", data)
                results[account_id] = [float(feat["f1"]), float(feat["f2"]), float(feat["f3"]), float(feat["f4"])]
                success = True
                break

    if not success:
        print(f"Could not get features for {account_id} from online store")

print(f"\nResults: {len(results)} entries")
print(results)

# COMMAND ----------
# If online store query failed, try the online table directly
if len(results) < len(lookup_keys):
    print("\nTrying online table lookup API...")
    for account_id in lookup_keys:
        if account_id in results:
            continue
        r = requests.get(
            f"{base_url}/api/2.1/unity-catalog/online-tables/{online_table_name_uc.replace('.', '/')}/lookup",
            headers=headers,
            params={"account_id": account_id}
        )
        print(f"Online table lookup {account_id} -> {r.status_code}: {r.text[:500]}")

# COMMAND ----------
# Write whatever results we have
if results:
    output = {"vectors": results}
    with open(output_path, "w") as fh:
        json.dump(output, fh)
    print(f"Results written to {output_path}")
    print(json.dumps(output, indent=2))
else:
    print("No results to write!")
