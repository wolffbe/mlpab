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

results = {}

# COMMAND ----------
# Step 1: Add a feature_vector column to the Delta table for VS index
print("=== Adding feature_vector column ===")
try:
    spark.sql(f"""
        ALTER TABLE {full_table_name}
        ADD COLUMN IF NOT EXISTS feature_vector ARRAY<DOUBLE>
    """)
    print("Column added")
except Exception as e:
    print(f"Column add error (may already exist): {e}")

# Update the feature_vector column
spark.sql(f"""
    UPDATE {full_table_name}
    SET feature_vector = ARRAY(f1, f2, f3, f4)
    WHERE feature_vector IS NULL
""")
print("feature_vector column populated")

# Verify
spark.sql(f"SELECT * FROM {full_table_name} LIMIT 3").show()

# COMMAND ----------
# Step 2: VS endpoint is already ONLINE (mlpabcef85c-vs-ep)
# Check status
ep_check = requests.get(f"{host}/api/2.0/vector-search/endpoints/{vs_endpoint_name}", headers=headers)
ep_state = ep_check.json().get("endpoint_status", {}).get("state", "UNKNOWN") if ep_check.status_code == 200 else "NOT_FOUND"
print(f"VS Endpoint state: {ep_state}")

# COMMAND ----------
# Step 3: Create a Delta Sync Vector Search Index
print("=== Creating Delta Sync VS Index ===")
# Check if index exists
idx_check = requests.get(f"{host}/api/2.0/vector-search/indexes/{vs_index_name}", headers=headers)
print(f"Index check: {idx_check.status_code}")

if idx_check.status_code != 200:
    idx_payload = {
        "name": vs_index_name,
        "endpoint_name": vs_endpoint_name,
        "primary_key": "account_id",
        "index_type": "DELTA_SYNC",
        "delta_sync_index_spec": {
            "source_table": full_table_name,
            "pipeline_type": "TRIGGERED",
            "embedding_vector_columns": [
                {
                    "name": "feature_vector",
                    "embedding_dimension": 4
                }
            ],
            "columns_to_sync": ["account_id", "f1", "f2", "f3", "f4"]
        }
    }
    idx_create = requests.post(f"{host}/api/2.0/vector-search/indexes", headers=headers, json=idx_payload)
    print(f"Index create: {idx_create.status_code} {idx_create.text[:1000]}")
    results["idx_create"] = {"status": idx_create.status_code, "text": idx_create.text[:500]}
else:
    idx_state = idx_check.json().get("status", {})
    print(f"Index exists: {json.dumps(idx_state)[:200]}")

# COMMAND ----------
# Step 4: Wait for the index to sync
max_wait = 300
start = time.time()
while time.time() - start < max_wait:
    time.sleep(15)
    idx_poll = requests.get(f"{host}/api/2.0/vector-search/indexes/{vs_index_name}", headers=headers)
    if idx_poll.status_code == 200:
        idx_data = idx_poll.json()
        idx_status = idx_data.get("status", {})
        idx_ready = idx_status.get("ready", False)
        idx_rows = idx_status.get("indexed_row_count", 0)
        print(f"[{int(time.time()-start)}s] Index ready={idx_ready}, rows={idx_rows}")
        if idx_ready and idx_rows > 0:
            print("Index is READY and has data!")
            results["idx_ready"] = True
            results["idx_rows"] = idx_rows
            break
        if idx_status.get("index_url"):
            print(f"Index URL: {idx_status.get('index_url')}")
    else:
        print(f"Poll error: {idx_poll.status_code} {idx_poll.text[:200]}")
        break

# COMMAND ----------
# Step 5: Query the index for each lookup key
print("=== Querying VS Index ===")
with open("/Workspace/Users/benedict@logicalclocks.com/mlpabcef85c/lookup_keys.txt") as f:
    lookup_keys = [line.strip() for line in f if line.strip()]

feature_results = {}
for key in lookup_keys:
    # Query by filter (exact key lookup)
    query_resp = requests.post(
        f"{host}/api/2.0/vector-search/indexes/{vs_index_name}/query",
        headers=headers,
        json={
            "num_results": 1,
            "columns": ["account_id", "f1", "f2", "f3", "f4"],
            "filters_json": json.dumps({"account_id": key}),
            "query_vector": [0.0, 0.0, 0.0, 0.0]
        }
    )
    if query_resp.status_code == 200:
        data = query_resp.json()
        cols = [c["name"] for c in data.get("manifest", {}).get("columns", [])]
        rows = data.get("result", {}).get("data_array", [])
        if rows:
            row_dict = dict(zip(cols, rows[0]))
            feature_results[key] = [row_dict.get("f1"), row_dict.get("f2"), row_dict.get("f3"), row_dict.get("f4")]
        else:
            print(f"Key {key}: no results returned")
    else:
        print(f"Key {key}: {query_resp.status_code} {query_resp.text[:200]}")
    results[f"query_{key}"] = query_resp.status_code if 'query_resp' in locals() else -1

print(f"Feature results count: {len(feature_results)}")
results["feature_results_count"] = len(feature_results)

# COMMAND ----------
# Save results
output = {"vectors": feature_results}
with open("/Workspace/Users/benedict@logicalclocks.com/mlpabcef85c/answers.json", "w") as f:
    json.dump(output, f)
print(f"Saved {len(feature_results)} results")
print(json.dumps(output, indent=2)[:1000])
dbutils.notebook.exit(json.dumps({"count": len(feature_results), "sample": dict(list(feature_results.items())[:3])}))
