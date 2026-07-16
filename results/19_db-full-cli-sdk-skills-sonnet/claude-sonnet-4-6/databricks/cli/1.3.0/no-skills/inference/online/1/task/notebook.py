# Databricks notebook source
# MAGIC %python

# COMMAND ----------
import json
import time
import requests

schema = "workspace.mlpabcef85c"
table_name = "profilesaa70e4"
full_table_name = f"{schema}.{table_name}"

# COMMAND ----------
# Read the CSV from workspace files
df = spark.read.option("header", "true").option("inferSchema", "true").csv(
    "file:/Workspace/Users/benedict@logicalclocks.com/mlpabcef85c/features.csv"
)
df.show()
df.printSchema()

# COMMAND ----------
# Create the Delta table (feature table) with record key account_id
spark.sql(f"DROP TABLE IF EXISTS {full_table_name}")
df.write.format("delta").mode("overwrite").saveAsTable(full_table_name)
print(f"Created table {full_table_name} with {df.count()} rows")

# COMMAND ----------
# Enable change data feed (required for online tables)
spark.sql(f"ALTER TABLE {full_table_name} SET TBLPROPERTIES ('delta.enableChangeDataFeed' = true)")
print("Enabled change data feed")

# COMMAND ----------
# Get token and host for API calls
ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
token = ctx.apiToken().get()
host = ctx.apiUrl().get()

headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

# COMMAND ----------
# Create an online table
online_table_name = f"{schema}.{table_name}"
online_table_spec = {
    "name": online_table_name,
    "spec": {
        "source_table_full_name": full_table_name,
        "primary_key_columns": ["account_id"],
        "run_triggered": {
            "triggered_type": "TRIGGERED"
        }
    }
}

# Check if exists, delete if so
resp = requests.get(
    f"{host}/api/2.0/online-tables/{online_table_name}",
    headers=headers
)
if resp.status_code == 200:
    print("Online table already exists, deleting...")
    del_resp = requests.delete(
        f"{host}/api/2.0/online-tables/{online_table_name}",
        headers=headers
    )
    print(f"Delete response: {del_resp.status_code}")
    time.sleep(10)

resp = requests.post(
    f"{host}/api/2.0/online-tables",
    headers=headers,
    json=online_table_spec
)
print(f"Create online table response: {resp.status_code}")
print(resp.text)

# COMMAND ----------
# Wait for online table to be ONLINE
for i in range(60):
    resp = requests.get(
        f"{host}/api/2.0/online-tables/{online_table_name}",
        headers=headers
    )
    data = resp.json()
    status = data.get("status", {})
    state = status.get("detailed_state", "UNKNOWN")
    print(f"Attempt {i+1}: state={state}")
    if state in ["ONLINE", "ONLINE_NO_PENDING_UPDATE"]:
        print("Online table is ONLINE!")
        break
    if "FAIL" in state or "ERROR" in state:
        print(f"Online table failed: {data}")
        break
    time.sleep(10)

# COMMAND ----------
# Read lookup keys
with open("/Workspace/Users/benedict@logicalclocks.com/mlpabcef85c/lookup_keys.txt") as f:
    lookup_keys = [line.strip() for line in f if line.strip()]
print(f"Lookup keys: {lookup_keys}")

# COMMAND ----------
# Query online table for each key
results = {}
for key in lookup_keys:
    resp = requests.get(
        f"{host}/api/2.0/online-tables/{online_table_name}/rows",
        headers=headers,
        params={"account_id": key}
    )
    if resp.status_code == 200:
        row_data = resp.json()
        print(f"Key {key}: {row_data}")
        if "columns" in row_data and "values" in row_data:
            cols = row_data["columns"]
            vals = row_data["values"][0] if row_data["values"] else None
            if vals:
                row_dict = dict(zip(cols, vals))
                results[key] = [row_dict["f1"], row_dict["f2"], row_dict["f3"], row_dict["f4"]]
    else:
        print(f"Key {key} failed: {resp.status_code} {resp.text}")

print(f"Results: {results}")

# COMMAND ----------
# Save results to workspace file
output = {"vectors": results}
output_path = "/Workspace/Users/benedict@logicalclocks.com/mlpabcef85c/answers.json"
with open(output_path, "w") as f:
    json.dump(output, f)
print(f"Saved results to {output_path}")
print(json.dumps(output, indent=2))
