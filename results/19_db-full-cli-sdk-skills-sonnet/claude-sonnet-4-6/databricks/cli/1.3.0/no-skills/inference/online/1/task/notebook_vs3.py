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
# Check what columns the table has
print("=== Table schema ===")
spark.sql(f"DESCRIBE {full_table_name}").show(truncate=False)
spark.sql(f"SELECT * FROM {full_table_name} LIMIT 3").show()

# COMMAND ----------
# Rebuild the table with feature_vector column
print("=== Rebuilding table with feature_vector ===")
# Create new version with feature_vector
spark.sql(f"""
CREATE OR REPLACE TABLE {full_table_name}
USING DELTA
TBLPROPERTIES ('delta.enableChangeDataFeed' = true)
AS
SELECT
    account_id,
    f1,
    f2,
    f3,
    f4,
    ARRAY(DOUBLE(f1), DOUBLE(f2), DOUBLE(f3), DOUBLE(f4)) AS feature_vector
FROM {full_table_name}
""")
print("Table rebuilt with feature_vector")
spark.sql(f"SELECT * FROM {full_table_name} LIMIT 3").show()

# COMMAND ----------
# VS endpoint is ONLINE
print("=== VS Endpoint Check ===")
ep_check = requests.get(f"{host}/api/2.0/vector-search/endpoints/{vs_endpoint_name}", headers=headers)
ep_state = ep_check.json().get("endpoint_status", {}).get("state", "UNKNOWN") if ep_check.status_code == 200 else "NOT_FOUND"
print(f"VS Endpoint state: {ep_state}")

# COMMAND ----------
# Create Delta Sync VS Index
print("=== Creating Delta Sync VS Index ===")
# Delete existing if any
idx_check = requests.get(f"{host}/api/2.0/vector-search/indexes/{vs_index_name}", headers=headers)
if idx_check.status_code == 200:
    print("Deleting existing index...")
    del_resp = requests.delete(f"{host}/api/2.0/vector-search/indexes/{vs_index_name}", headers=headers)
    print(f"Delete: {del_resp.status_code}")
    time.sleep(5)

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
print(f"Index create: {idx_create.status_code}")
print(idx_create.text[:2000])
results["idx_create"] = {"status": idx_create.status_code, "text": idx_create.text[:500]}

# COMMAND ----------
# Wait for the index to sync
print("=== Waiting for VS Index sync ===")
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
        print(f"[{int(time.time()-start)}s] ready={idx_ready}, rows={idx_rows}")
        print(f"Full status: {json.dumps(idx_status)[:300]}")
        if idx_ready and idx_rows > 0:
            print("Index is READY!")
            results["idx_ready"] = True
            results["idx_rows"] = idx_rows
            break
    else:
        print(f"Poll error: {idx_poll.status_code} {idx_poll.text[:200]}")
        break

# COMMAND ----------
# Query the index for each lookup key using filter
print("=== Querying VS Index by filter ===")
with open("/Workspace/Users/benedict@logicalclocks.com/mlpabcef85c/lookup_keys.txt") as f:
    lookup_keys = [line.strip() for line in f if line.strip()]

feature_results = {}
for key in lookup_keys:
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
            print(f"Key {key}: no results. Full response: {json.dumps(data)[:200]}")
    else:
        print(f"Key {key}: {query_resp.status_code} {query_resp.text[:200]}")

print(f"Feature results: {len(feature_results)}")

# COMMAND ----------
# Try scan-based lookup (alternative approach)
if len(feature_results) < len(lookup_keys):
    print("=== Trying scan-based lookup ===")
    scan_resp = requests.post(
        f"{host}/api/2.0/vector-search/indexes/{vs_index_name}/scan",
        headers=headers,
        json={
            "num_results": 200,
            "columns": ["account_id", "f1", "f2", "f3", "f4"]
        }
    )
    print(f"Scan: {scan_resp.status_code} {scan_resp.text[:500]}")
    if scan_resp.status_code == 200:
        scan_data = scan_resp.json()
        cols = [c["name"] for c in scan_data.get("manifest", {}).get("columns", [])]
        rows = scan_data.get("result", {}).get("data_array", [])
        all_rows = {dict(zip(cols, row))["account_id"]: dict(zip(cols, row)) for row in rows}
        for key in lookup_keys:
            if key in all_rows and key not in feature_results:
                row = all_rows[key]
                feature_results[key] = [row.get("f1"), row.get("f2"), row.get("f3"), row.get("f4")]

print(f"Final feature results: {len(feature_results)}")

# COMMAND ----------
# Save results
output = {"vectors": feature_results}
with open("/Workspace/Users/benedict@logicalclocks.com/mlpabcef85c/answers.json", "w") as f:
    json.dump(output, f)
print(json.dumps(output, indent=2)[:1000])
dbutils.notebook.exit(json.dumps({"count": len(feature_results), "results_sample": dict(list(feature_results.items())[:3])}))
