# Databricks notebook source
import json, time, requests

schema = "workspace.mlpabcef85c"
table_name = "profilesaa70e4"
full_table_name = f"{schema}.{table_name}"
prefix = "mlpabcef85c"
vs_endpoint_name = f"{prefix}-vs-ep"
vs_idx_da = f"{schema}.{table_name}_da_idx"

ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
token = ctx.apiToken().get()
host = ctx.apiUrl().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

results = {}

# COMMAND ----------
# Create a Direct Access Vector Search Index
print("=== Creating Direct Access VS Index ===")
# Delete existing if any
idx_check = requests.get(f"{host}/api/2.0/vector-search/indexes/{vs_idx_da}", headers=headers)
print(f"Existing DA index check: {idx_check.status_code}")
if idx_check.status_code == 200:
    print("Deleting existing DA index...")
    del_resp = requests.delete(f"{host}/api/2.0/vector-search/indexes/{vs_idx_da}", headers=headers)
    print(f"Delete: {del_resp.status_code}")
    time.sleep(5)

# Create the Direct Access index with embedding vector column
idx_payload = {
    "name": vs_idx_da,
    "endpoint_name": vs_endpoint_name,
    "primary_key": "account_id",
    "index_type": "DIRECT_ACCESS",
    "direct_access_index_spec": {
        "schema_json": json.dumps({
            "account_id": "string",
            "f1": "float",
            "f2": "float",
            "f3": "float",
            "f4": "float",
            "feature_vector": "array<float>"
        }),
        "embedding_vector_columns": [
            {
                "name": "feature_vector",
                "embedding_dimension": 4
            }
        ]
    }
}
print(f"Creating index with payload: {json.dumps(idx_payload, indent=2)}")
idx_create = requests.post(f"{host}/api/2.0/vector-search/indexes", headers=headers, json=idx_payload)
print(f"Index create: {idx_create.status_code}")
print(idx_create.text[:2000])
results["idx_create"] = {"status": idx_create.status_code, "text": idx_create.text[:500]}

# COMMAND ----------
# Wait for the Direct Access index to be ready
print("=== Waiting for DA Index ===")
for i in range(20):
    time.sleep(5)
    idx_poll = requests.get(f"{host}/api/2.0/vector-search/indexes/{vs_idx_da}", headers=headers)
    if idx_poll.status_code == 200:
        idx_data = idx_poll.json()
        idx_ready = idx_data.get("status", {}).get("ready", False)
        idx_msg = idx_data.get("status", {}).get("message", "")
        print(f"[{(i+1)*5}s] ready={idx_ready}, msg={idx_msg}")
        if idx_ready:
            print("Index is READY!")
            results["idx_ready"] = True
            break
    else:
        print(f"Poll error: {idx_poll.status_code} {idx_poll.text[:200]}")
        break

# COMMAND ----------
# Load feature data from Delta table
print("=== Loading feature data ===")
df = spark.table(full_table_name)
feature_data = []
for row in df.collect():
    feature_data.append({
        "account_id": str(row["account_id"]),
        "f1": float(row["f1"]),
        "f2": float(row["f2"]),
        "f3": float(row["f3"]),
        "f4": float(row["f4"]),
        "feature_vector": [float(row["f1"]), float(row["f2"]), float(row["f3"]), float(row["f4"])]
    })
print(f"Loaded {len(feature_data)} records")
print(f"Sample: {feature_data[:2]}")

# COMMAND ----------
# Upsert data into the Direct Access index
print("=== Upserting data ===")
upsert_payload = {"inputs_json": json.dumps(feature_data)}
upsert_resp = requests.post(
    f"{host}/api/2.0/vector-search/indexes/{vs_idx_da}/upsert-data",
    headers=headers,
    json=upsert_payload
)
print(f"Upsert: {upsert_resp.status_code} {upsert_resp.text[:500]}")
results["upsert"] = {"status": upsert_resp.status_code, "text": upsert_resp.text[:300]}

# COMMAND ----------
# Wait for data to be indexed
print("=== Waiting for data to be indexed ===")
for i in range(10):
    time.sleep(10)
    idx_poll = requests.get(f"{host}/api/2.0/vector-search/indexes/{vs_idx_da}", headers=headers)
    if idx_poll.status_code == 200:
        idx_data = idx_poll.json()
        idx_rows = idx_data.get("status", {}).get("indexed_row_count", 0)
        print(f"[{(i+1)*10}s] indexed_rows={idx_rows}")
        if idx_rows > 0:
            print(f"Data indexed! {idx_rows} rows")
            break
    else:
        break

# COMMAND ----------
# Query the index for each lookup key
print("=== Querying Direct Access VS Index ===")
with open("/Workspace/Users/benedict@logicalclocks.com/mlpabcef85c/lookup_keys.txt") as f:
    lookup_keys = [line.strip() for line in f if line.strip()]

feature_results = {}

# Try scan-based approach first
scan_resp = requests.post(
    f"{host}/api/2.0/vector-search/indexes/{vs_idx_da}/scan",
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
    print(f"Scan cols: {cols}, rows count: {len(rows)}")
    all_rows = {}
    for row in rows:
        row_dict = dict(zip(cols, row))
        aid = row_dict.get("account_id")
        all_rows[aid] = row_dict

    for key in lookup_keys:
        if key in all_rows:
            row = all_rows[key]
            feature_results[key] = [row.get("f1"), row.get("f2"), row.get("f3"), row.get("f4")]
        else:
            # Try direct query
            q_resp = requests.post(
                f"{host}/api/2.0/vector-search/indexes/{vs_idx_da}/query",
                headers=headers,
                json={
                    "num_results": 1,
                    "columns": ["account_id", "f1", "f2", "f3", "f4"],
                    "filters_json": json.dumps({"account_id": key}),
                    "query_vector": [0.0, 0.0, 0.0, 0.0]
                }
            )
            if q_resp.status_code == 200:
                q_data = q_resp.json()
                q_cols = [c["name"] for c in q_data.get("manifest", {}).get("columns", [])]
                q_rows = q_data.get("result", {}).get("data_array", [])
                if q_rows:
                    q_row = dict(zip(q_cols, q_rows[0]))
                    feature_results[key] = [q_row.get("f1"), q_row.get("f2"), q_row.get("f3"), q_row.get("f4")]

print(f"Feature results count: {len(feature_results)}")
print(json.dumps(feature_results, indent=2)[:1000])

# COMMAND ----------
# Save results
output = {"vectors": feature_results}
with open("/Workspace/Users/benedict@logicalclocks.com/mlpabcef85c/answers.json", "w") as f:
    json.dump(output, f)
print(f"Saved {len(feature_results)} results")
dbutils.notebook.exit(json.dumps({"count": len(feature_results), "sample": dict(list(feature_results.items())[:3]), "upsert_status": results.get("upsert", {}).get("status")}))
