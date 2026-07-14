# Databricks notebook source
import json, time, requests

schema = "workspace.mlpabcef85c"
table_name = "profilesaa70e4"
prefix = "mlpabcef85c"
vs_idx_da = f"{schema}.{table_name}_da_idx"

ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
token = ctx.apiToken().get()
host = ctx.apiUrl().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

# COMMAND ----------
# Query the VS index for A0015 specifically
print("=== Querying A0015 ===")
q_resp = requests.post(
    f"{host}/api/2.0/vector-search/indexes/{vs_idx_da}/query",
    headers=headers,
    json={
        "num_results": 1,
        "columns": ["account_id", "f1", "f2", "f3", "f4"],
        "filters_json": json.dumps({"account_id": "A0015"}),
        "query_vector": [0.0493, -1.6131, 1.7496, 3.831]
    }
)
print(f"Query A0015: {q_resp.status_code} {q_resp.text[:500]}")

# COMMAND ----------
# Try scan to find A0015
scan_resp = requests.post(
    f"{host}/api/2.0/vector-search/indexes/{vs_idx_da}/scan",
    headers=headers,
    json={
        "num_results": 200,
        "columns": ["account_id", "f1", "f2", "f3", "f4"],
        "last_primary_key": "A0010"  # Start from A0010 to find A0015
    }
)
print(f"Scan from A0010: {scan_resp.status_code} {scan_resp.text[:1000]}")

# COMMAND ----------
# Load current answers and add A0015
with open("/Workspace/Users/benedict@logicalclocks.com/mlpabcef85c/answers.json") as f:
    current = json.load(f)

print(f"Current count: {len(current.get('vectors', {}))}")
print(f"A0015 in current: {'A0015' in current.get('vectors', {})}")

# Add A0015 from the VS index query if found
if q_resp.status_code == 200:
    q_data = q_resp.json()
    cols = [c["name"] for c in q_data.get("manifest", {}).get("columns", [])]
    rows = q_data.get("result", {}).get("data_array", [])
    if rows:
        row_dict = dict(zip(cols, rows[0]))
        if row_dict.get("account_id") == "A0015":
            current["vectors"]["A0015"] = [row_dict.get("f1"), row_dict.get("f2"), row_dict.get("f3"), row_dict.get("f4")]
            print(f"Added A0015 from query: {current['vectors']['A0015']}")

# If scan found A0015
if scan_resp.status_code == 200:
    scan_data = scan_resp.json()
    cols = [c["name"] for c in scan_data.get("manifest", {}).get("columns", [])]
    rows = scan_data.get("result", {}).get("data_array", [])
    for row in rows:
        row_dict = dict(zip(cols, row))
        if row_dict.get("account_id") == "A0015":
            current["vectors"]["A0015"] = [row_dict.get("f1"), row_dict.get("f2"), row_dict.get("f3"), row_dict.get("f4")]
            print(f"Added A0015 from scan: {current['vectors']['A0015']}")
            break

print(f"Final count: {len(current.get('vectors', {}))}")
print(f"A0015: {current['vectors'].get('A0015')}")

# Save updated results
with open("/Workspace/Users/benedict@logicalclocks.com/mlpabcef85c/answers.json", "w") as f:
    json.dump(current, f)
print("Saved!")
dbutils.notebook.exit(json.dumps({"count": len(current.get("vectors", {})), "has_A0015": "A0015" in current.get("vectors", {}), "A0015_val": current["vectors"].get("A0015")}))
