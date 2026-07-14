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
online_table_name = "profilesaa70e4_online"
online_table_full = f"{schema}.{online_table_name}"
output_path = "/Volumes/workspace/mlpabae7d2f/mlpabae7d2f_vol/answers.json"

lookup_keys = [
    "A0003", "A0005", "A0012", "A0015", "A0023", "A0030", "A0031", "A0034",
    "A0048", "A0049", "A0055", "A0063", "A0066", "A0072", "A0085", "A0090",
    "A0103", "A0109", "A0112", "A0113"
]

# COMMAND ----------
# Check what the online store looks like
r = requests.get(f"{base_url}/api/2.0/feature-store/online-stores/{online_store_name}", headers=headers)
print(f"Online store: {r.status_code}: {r.text[:500]}")

# Check feature tables in the online store
r2 = requests.get(f"{base_url}/api/2.0/feature-store/online-stores/{online_store_name}/feature-tables", headers=headers)
print(f"Feature tables: {r2.status_code}: {r2.text[:1000]}")

# COMMAND ----------
# Try querying the online store with different paths
test_key = "A0003"

paths = [
    f"/api/2.0/feature-store/online-stores/{online_store_name}/feature-tables/{table_name}/lookup",
    f"/api/2.0/feature-store/online-stores/{online_store_name}/lookup",
    f"/api/2.0/feature-store/online-tables/{online_table_full}/lookup",
    f"/api/2.0/feature-store/online-tables/{online_table_name}/lookup",
]

for path in paths:
    r = requests.get(f"{base_url}{path}", headers=headers, params={"account_id": test_key})
    print(f"GET {path} -> {r.status_code}: {r.text[:400]}")
    r = requests.post(f"{base_url}{path}", headers=headers, json={"account_id": test_key})
    print(f"POST {path} -> {r.status_code}: {r.text[:400]}")

# COMMAND ----------
# Check if online table was created (UC online tables)
r = requests.get(f"{base_url}/api/2.1/unity-catalog/online-tables/{online_table_full}", headers=headers)
print(f"UC online table: {r.status_code}: {r.text[:1000]}")

# COMMAND ----------
# Try serving endpoint approach
# The online store might expose a model serving endpoint
r = requests.get(f"{base_url}/api/2.0/serving-endpoints", headers=headers)
print(f"Serving endpoints: {r.status_code}: {r.text[:1000]}")

# COMMAND ----------
# Try the feature serving API
r = requests.post(
    f"{base_url}/api/2.0/feature-store/online-stores/{online_store_name}/feature-tables/{table_name}/lookup",
    headers=headers,
    json={"lookup_key": {"account_id": test_key}}
)
print(f"Feature lookup: {r.status_code}: {r.text[:1000]}")

# COMMAND ----------
# Try reading directly from the online table using Spark (if it's a Delta table)
try:
    df_online = spark.table(online_table_full)
    print(f"Online table schema: {df_online.schema}")
    df_online.filter(df_online.account_id.isin(lookup_keys)).show(5)
except Exception as e:
    print(f"Cannot read online table as Spark table: {e}")

# COMMAND ----------
# Try the serving endpoint for the online store itself
# The online store might have its own REST API endpoint
for endpoint_format in [
    f"https://{online_store_name}.{host.split('dbc-')[1].split('.')[0]}.cloud.databricks.com",
    f"{base_url}/api/2.0/feature-serving/",
]:
    try:
        r = requests.get(endpoint_format, headers=headers, timeout=5)
        print(f"Endpoint {endpoint_format}: {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"Endpoint {endpoint_format}: ERROR: {e}")

dbutils.notebook.exit("diagnostic_complete")
