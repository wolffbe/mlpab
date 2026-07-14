# Databricks notebook source
import json
import time
import requests

schema = "workspace.mlpabcef85c"
table_name = "profilesaa70e4"
full_table_name = f"{schema}.{table_name}"

# COMMAND ----------
# Get token and host for API calls
ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
token = ctx.apiToken().get()
host = ctx.apiUrl().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
print(f"Host: {host}")

# COMMAND ----------
# Check current online tables
resp = requests.get(f"{host}/api/2.0/online-tables", headers=headers)
print(f"List online tables: {resp.status_code}")
print(resp.text[:2000])

# COMMAND ----------
# Check if source table has CDF enabled
result = spark.sql(f"DESCRIBE DETAIL {full_table_name}").collect()
for row in result:
    print(row)

spark.sql(f"SHOW TBLPROPERTIES {full_table_name}").show(truncate=False)

# COMMAND ----------
# Create the online table
online_table_spec = {
    "name": full_table_name,
    "spec": {
        "source_table_full_name": full_table_name,
        "primary_key_columns": ["account_id"],
        "run_triggered": {}
    }
}

print("Creating online table...")
resp = requests.post(
    f"{host}/api/2.0/online-tables",
    headers=headers,
    json=online_table_spec
)
print(f"Status: {resp.status_code}")
print(resp.text[:3000])

# COMMAND ----------
# Wait for the online table to be ONLINE
online_table_name = full_table_name
max_wait = 120
start = time.time()
while time.time() - start < max_wait:
    resp = requests.get(
        f"{host}/api/2.0/online-tables/{online_table_name}",
        headers=headers
    )
    data = resp.json()
    print(f"Status: {resp.status_code}")
    print(json.dumps(data, indent=2)[:2000])

    status = data.get("status", {})
    state = status.get("detailed_state", "")
    print(f"State: {state}")

    if state in ["ONLINE", "ONLINE_NO_PENDING_UPDATE"]:
        print("Online table is ready!")
        break
    if "FAIL" in str(state).upper() or "ERROR" in str(state).upper():
        print(f"FAILED: {data}")
        break

    time.sleep(15)

# COMMAND ----------
# Read lookup keys
with open("/Workspace/Users/benedict@logicalclocks.com/mlpabcef85c/lookup_keys.txt") as f:
    lookup_keys = [line.strip() for line in f if line.strip()]
print(f"Lookup keys count: {len(lookup_keys)}")
print(lookup_keys)

# COMMAND ----------
# Query the online table using the serving endpoint
# First check the online table details to find query endpoint
resp = requests.get(
    f"{host}/api/2.0/online-tables/{online_table_name}",
    headers=headers
)
print(f"Online table details: {resp.status_code}")
ot_data = resp.json()
print(json.dumps(ot_data, indent=2)[:3000])

# COMMAND ----------
# Try querying the online table via the feature serving endpoint
# The online table can be queried via /api/2.0/serving-endpoints/{name}/invocations
# or via the feature lookup API

# Check if there's a feature serving endpoint
resp = requests.get(f"{host}/api/2.0/serving-endpoints", headers=headers)
print(f"Serving endpoints: {resp.status_code}")
endpoints_data = resp.json()
print(json.dumps(endpoints_data, indent=2)[:2000])

# COMMAND ----------
# Query online table directly via API
results = {}
for key in lookup_keys:
    resp = requests.get(
        f"{host}/api/2.0/online-tables/{online_table_name}/rows",
        headers=headers,
        params={"account_id": key}
    )
    print(f"Key {key}: status={resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"  Data: {json.dumps(data)[:500]}")
    else:
        print(f"  Error: {resp.text[:200]}")

# COMMAND ----------
# Try looking up via feature store client
from databricks.feature_store import FeatureStoreClient
fs = FeatureStoreClient()
print("FeatureStoreClient created")

# Check online store access
try:
    # Use the online store for lookup
    import pandas as pd
    lookup_df = pd.DataFrame({"account_id": lookup_keys})

    # Create feature lookups
    from databricks.feature_store import FeatureLookup

    feature_lookups = [
        FeatureLookup(
            table_name=full_table_name,
            lookup_key="account_id",
            feature_names=["f1", "f2", "f3", "f4"]
        )
    ]

    # Get online features
    scored_df = fs.score_batch(
        model_uri=None,
        df=spark.createDataFrame(lookup_df),
        result_type="double",
        feature_lookups=feature_lookups
    )
    scored_df.show()
except Exception as e:
    print(f"FeatureStore approach failed: {e}")

# COMMAND ----------
# Direct SQL read from the online table (Databricks feature serving)
# Alternative: use the REST API for feature lookup
results = {}
for key in lookup_keys:
    try:
        # Try using the online table REST API
        resp = requests.post(
            f"{host}/api/2.0/online-tables/{online_table_name}/rows",
            headers=headers,
            json={"account_id": key}
        )
        print(f"POST Key {key}: status={resp.status_code}, text={resp.text[:200]}")
    except Exception as e:
        print(f"Error for {key}: {e}")

# COMMAND ----------
# Save whatever results we have
output = {"vectors": results}
output_path = "/Workspace/Users/benedict@logicalclocks.com/mlpabcef85c/answers.json"
with open(output_path, "w") as f:
    json.dump(output, f)
print(f"Saved results: {json.dumps(output, indent=2)[:1000]}")
