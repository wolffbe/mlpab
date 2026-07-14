# Databricks notebook source
import json, time, requests

schema = "workspace.mlpabcef85c"
table_name = "profilesaa70e4"
full_table_name = f"{schema}.{table_name}"
prefix = "mlpabcef85c"
vs_endpoint_name = f"{prefix}-vs-ep"
vs_index_name = f"{schema}.{table_name}_vs_idx"

ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
token = ctx.apiToken().get()
host = ctx.apiUrl().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

results = {}

# COMMAND ----------
# Step 1: Check Vector Search endpoint
print("=== Vector Search Endpoint ===")
vs_ep_check = requests.get(f"{host}/api/2.0/vector-search/endpoints/{vs_endpoint_name}", headers=headers)
print(f"Endpoint check: {vs_ep_check.status_code} {vs_ep_check.text[:300]}")

if vs_ep_check.status_code != 200:
    # Create the Vector Search endpoint
    vs_ep_payload = {"name": vs_endpoint_name, "endpoint_type": "STANDARD"}
    vs_ep_create = requests.post(
        f"{host}/api/2.0/vector-search/endpoints",
        headers=headers,
        json=vs_ep_payload
    )
    print(f"Endpoint create: {vs_ep_create.status_code} {vs_ep_create.text[:500]}")
    results["ep_create"] = {"status": vs_ep_create.status_code, "text": vs_ep_create.text[:300]}
else:
    ep_state = vs_ep_check.json().get("endpoint_status", {}).get("state", "UNKNOWN")
    print(f"Endpoint exists, state: {ep_state}")
    results["ep_state"] = ep_state

# COMMAND ----------
# Wait for Vector Search endpoint to be ONLINE
max_wait = 300
start = time.time()
while time.time() - start < max_wait:
    ep_poll = requests.get(f"{host}/api/2.0/vector-search/endpoints/{vs_endpoint_name}", headers=headers)
    if ep_poll.status_code == 200:
        ep_data = ep_poll.json()
        ep_state = ep_data.get("endpoint_status", {}).get("state", "UNKNOWN")
        print(f"[{int(time.time()-start)}s] VS Endpoint state: {ep_state}")
        if ep_state == "ONLINE":
            print("VS Endpoint is ONLINE!")
            results["ep_final_state"] = ep_state
            break
        if "FAIL" in str(ep_state).upper():
            print(f"VS Endpoint FAILED: {json.dumps(ep_data)[:500]}")
            results["ep_failed"] = ep_state
            break
    else:
        print(f"Poll error: {ep_poll.status_code}")
        break
    time.sleep(15)

# COMMAND ----------
# Step 2: Create Direct Access Vector Search Index
print("=== Vector Search Index (Direct Access) ===")
vs_idx_check = requests.get(f"{host}/api/2.0/vector-search/indexes/{vs_index_name}", headers=headers)
print(f"Index check: {vs_idx_check.status_code} {vs_idx_check.text[:300]}")

if vs_idx_check.status_code != 200:
    # Create a DIRECT_ACCESS index (no embedding needed)
    vs_idx_payload = {
        "name": vs_index_name,
        "endpoint_name": vs_endpoint_name,
        "primary_key": "account_id",
        "index_type": "DIRECT_ACCESS",
        "direct_access_index_spec": {
            "schema_json": json.dumps({
                "account_id": {"type": "string"},
                "f1": {"type": "float"},
                "f2": {"type": "float"},
                "f3": {"type": "float"},
                "f4": {"type": "float"}
            }),
            "embedding_source_columns": []
        }
    }
    vs_idx_create = requests.post(
        f"{host}/api/2.0/vector-search/indexes",
        headers=headers,
        json=vs_idx_payload
    )
    print(f"Index create: {vs_idx_create.status_code} {vs_idx_create.text[:1000]}")
    results["idx_create"] = {"status": vs_idx_create.status_code, "text": vs_idx_create.text[:500]}
else:
    idx_state = vs_idx_check.json().get("status", {}).get("indexed_row_count", 0)
    print(f"Index exists, rows: {idx_state}")

# COMMAND ----------
# Wait for index to be ONLINE
for i in range(20):
    time.sleep(10)
    idx_poll = requests.get(f"{host}/api/2.0/vector-search/indexes/{vs_index_name}", headers=headers)
    if idx_poll.status_code == 200:
        idx_data = idx_poll.json()
        idx_status = idx_data.get("status", {})
        print(f"[{(i+1)*10}s] Index status: {json.dumps(idx_status)[:200]}")
        if idx_status.get("ready", False):
            print("Index is READY!")
            break
    else:
        print(f"Index poll error: {idx_poll.status_code} {idx_poll.text[:200]}")
        break

# COMMAND ----------
# Step 3: Upsert feature data into the index
print("=== Upserting Feature Data ===")
df = spark.table(full_table_name).toPandas()
records = []
for _, row in df.iterrows():
    records.append({
        "account_id": str(row['account_id']),
        "f1": float(row['f1']),
        "f2": float(row['f2']),
        "f3": float(row['f3']),
        "f4": float(row['f4'])
    })
print(f"Upserting {len(records)} records...")

# Upsert in batches of 100
batch_size = 100
for i in range(0, len(records), batch_size):
    batch = records[i:i+batch_size]
    upsert_payload = {"inputs_json": json.dumps(batch)}
    upsert_resp = requests.post(
        f"{host}/api/2.0/vector-search/indexes/{vs_index_name}/upsert-data",
        headers=headers,
        json=upsert_payload
    )
    print(f"Upsert batch {i//batch_size + 1}: {upsert_resp.status_code} {upsert_resp.text[:200]}")
    results[f"upsert_{i}"] = upsert_resp.status_code

# COMMAND ----------
# Step 4: Query the index for each lookup key
print("=== Querying Vector Search Index ===")
with open("/Workspace/Users/benedict@logicalclocks.com/mlpabcef85c/lookup_keys.txt") as f:
    lookup_keys = [line.strip() for line in f if line.strip()]

feature_results = {}
for key in lookup_keys:
    # Use the lookup endpoint (direct key access)
    lookup_resp = requests.post(
        f"{host}/api/2.0/vector-search/indexes/{vs_index_name}/query",
        headers=headers,
        json={
            "num_results": 1,
            "columns": ["account_id", "f1", "f2", "f3", "f4"],
            "filters_json": json.dumps({"account_id": key}),
            "query_type": "ANN"
        }
    )
    if lookup_resp.status_code == 200:
        data = lookup_resp.json()
        cols = data.get("manifest", {}).get("columns", [])
        rows = data.get("result", {}).get("data_array", [])
        if rows:
            row_dict = {c["name"]: v for c, v in zip(cols, rows[0])}
            feature_results[key] = [row_dict.get("f1"), row_dict.get("f2"), row_dict.get("f3"), row_dict.get("f4")]
        else:
            print(f"Key {key}: no results")
    else:
        print(f"Key {key}: {lookup_resp.status_code} {lookup_resp.text[:200]}")

print(f"Feature results count: {len(feature_results)}")
results["feature_results_count"] = len(feature_results)

# COMMAND ----------
# Save results
output = {"vectors": feature_results}
with open("/Workspace/Users/benedict@logicalclocks.com/mlpabcef85c/answers.json", "w") as f:
    json.dump(output, f)
print(f"Saved: {json.dumps(output, indent=2)[:1000]}")
dbutils.notebook.exit(json.dumps({"count": len(feature_results)}))
