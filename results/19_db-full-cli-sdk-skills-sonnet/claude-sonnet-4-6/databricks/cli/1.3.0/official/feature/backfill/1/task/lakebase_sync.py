# Databricks notebook source
# Explore Lakebase and synced tables for online access

# COMMAND ----------
import requests, json
host = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

results = []

# Check lakebase API
paths = [
    "/api/2.0/lakebase/synced-tables",
    "/api/2.0/lakebase/sync",
    "/api/2.0/lakebase/tables",
    "/api/2.0/lakebase/instances",
    "/api/2.0/lakebase/databases",
    "/api/beta/catalog/synced-tables",
    "/api/2.0/catalog/synced-tables",
]

for path in paths:
    r = requests.get(f"https://{host}{path}", headers=headers)
    results.append(f"GET {path}: {r.status_code} {r.text[:150]}")

print('\n'.join(results))
spark.createDataFrame([(r,) for r in results], ["result"]).write.mode("overwrite").saveAsTable("workspace.mlpab0442b8.lakebase_discovery")

# COMMAND ----------
results2 = []

# Try the feature-store publish with a workaround
# The GetCatalog error might be because the catalog "workspace" is not in the old format
# Try registering the table in feature store first

# Check existing feature store tables
r_tables = requests.get(f"https://{host}/api/2.0/feature-store/feature-tables", headers=headers)
results2.append(f"GET feature-tables: {r_tables.status_code} {r_tables.text[:400]}")

# Get feature store table details
r_ft = requests.get(f"https://{host}/api/2.0/feature-store/feature-tables/workspace.mlpab0442b8.accountse81ff1", headers=headers)
results2.append(f"GET feature table: {r_ft.status_code} {r_ft.text[:400]}")

print('\n'.join(results2))
spark.createDataFrame([(r,) for r in results2], ["result"]).write.mode("append").saveAsTable("workspace.mlpab0442b8.lakebase_discovery")

# COMMAND ----------
results3 = []

# Try to check if feature serving endpoint has feature_serving_spec
r_ep = requests.get(f"https://{host}/api/2.0/serving-endpoints/mlpab0442b8-accountse81ff1", headers=headers)
ep_json = r_ep.json()
results3.append(f"Endpoint keys: {list(ep_json.keys())}")
results3.append(f"Full: {json.dumps(ep_json)[:600]}")

# Try to get the endpoint's feature spec separately
r_spec = requests.get(f"https://{host}/api/2.0/serving-endpoints/mlpab0442b8-accountse81ff1/feature-spec", headers=headers)
results3.append(f"Feature spec endpoint: {r_spec.status_code} {r_spec.text[:200]}")

print('\n'.join(results3))
spark.createDataFrame([(r,) for r in results3], ["result"]).write.mode("append").saveAsTable("workspace.mlpab0442b8.lakebase_discovery")
