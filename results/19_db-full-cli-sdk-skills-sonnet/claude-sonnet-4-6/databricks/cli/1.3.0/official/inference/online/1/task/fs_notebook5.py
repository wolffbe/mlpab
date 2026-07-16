# Databricks notebook source
# COMMAND ----------
import json, time, sys
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

results_log = {}

# COMMAND ----------
# Check available Python packages
print("Python version:", sys.version)
print("Checking feature engineering packages...")
for pkg in ["databricks.feature_engineering", "databricks.feature_store", "databricks.sdk"]:
    try:
        __import__(pkg)
        print(f"  {pkg}: AVAILABLE")
    except ImportError as e:
        print(f"  {pkg}: NOT AVAILABLE ({e})")

# COMMAND ----------
# Try Unity Catalog Feature Engineering REST API (2.1)
uc_api_paths = [
    f"/api/2.1/unity-catalog/feature-tables/{table_name}",
    f"/api/2.1/unity-catalog/feature-tables",
    f"/api/2.0/feature-store/tables/{table_name}",
]

for path in uc_api_paths:
    r = requests.get(f"{base_url}{path}", headers=headers)
    print(f"GET {path} -> {r.status_code}: {r.text[:300]}")

# COMMAND ----------
# Try to create a UC feature table
create_body = {
    "name": table_name,
    "primary_keys": [{"name": "account_id"}],
}
r = requests.post(f"{base_url}/api/2.1/unity-catalog/feature-tables", headers=headers, json=create_body)
print(f"POST /api/2.1/unity-catalog/feature-tables -> {r.status_code}: {r.text[:1000]}")

# COMMAND ----------
# Try to use Feature Engineering SDK if available
try:
    from databricks.feature_engineering import FeatureEngineeringClient
    print("FeatureEngineeringClient available!")
    fe = FeatureEngineeringClient()

    # Create feature table
    df = spark.table(table_name)
    try:
        fe.drop_table(name=table_name)
    except:
        pass

    fe.create_table(
        name=table_name,
        primary_keys=["account_id"],
        df=df,
        description="Account feature profiles v1"
    )
    print("Feature table created with FE client!")
    results_log["fe_create"] = "success"
except ImportError:
    print("FeatureEngineeringClient not available")
    results_log["fe_create"] = "not_available"

# COMMAND ----------
# Now try publishing with the CLI-compatible API using SNAPSHOT mode
publish_body = {
    "publish_spec": {
        "online_store": online_store_name,
        "online_table_name": online_table_name_uc,
        "publish_mode": "SNAPSHOT"
    }
}
r = requests.post(f"{base_url}/api/2.0/feature-store/tables/{table_name}/publish", headers=headers, json=publish_body)
print(f"Publish SNAPSHOT -> {r.status_code}: {r.text[:2000]}")
results_log["publish"] = {"status": r.status_code, "response": r.text[:500]}

# COMMAND ----------
# If publish worked, wait and query online store
if r.status_code in (200, 201, 202):
    print("Publish succeeded, waiting 30s...")
    time.sleep(30)

    # Try various online lookup paths
    test_key = "A0003"

    lookup_paths = [
        f"/api/2.0/feature-store/online-stores/{online_store_name}/feature-tables/{table_name}/lookup",
        f"/api/2.0/feature-store/tables/{table_name}/online-lookup",
        f"/api/2.0/feature-store/online/{table_name}",
        f"/api/2.0/serving-endpoints/{online_store_name}/invocations",
    ]

    for path in lookup_paths:
        r = requests.get(f"{base_url}{path}", headers=headers, params={"account_id": test_key})
        print(f"GET {path} -> {r.status_code}: {r.text[:400]}")
        r2 = requests.post(f"{base_url}{path}", headers=headers, json={"account_id": test_key})
        print(f"POST {path} -> {r2.status_code}: {r2.text[:400]}")

# COMMAND ----------
# Last resort: read from Delta directly using SQL (offline, but for verification)
print("\n--- Reading features from Delta table (offline for comparison) ---")
df = spark.table(table_name)
df.show(5)

# Collect features for lookup keys into memory
rows = df.filter(df.account_id.isin(lookup_keys)).collect()
print(f"Found {len(rows)} rows for {len(lookup_keys)} lookup keys")

# Store these for potential fallback (but this is offline read)
fallback_results = {}
for row in rows:
    fallback_results[row.account_id] = [row.f1, row.f2, row.f3, row.f4]

print(f"Fallback results: {json.dumps(fallback_results)}")
results_log["fallback"] = fallback_results

dbutils.notebook.exit(json.dumps(results_log))
