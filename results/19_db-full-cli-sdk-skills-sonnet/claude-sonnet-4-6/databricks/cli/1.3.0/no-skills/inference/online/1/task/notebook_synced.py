# Databricks notebook source
import json
import time
import requests

schema = "workspace.mlpabcef85c"
table_name = "profilesaa70e4"
full_table_name = f"{schema}.{table_name}"
prefix = "mlpabcef85c"

ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
token = ctx.apiToken().get()
host = ctx.apiUrl().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
print(f"Host: {host}")

# COMMAND ----------
# Try to find the synced tables API
print("=== Discovering Synced Tables API ===")
for path in [
    "/api/2.0/synced-tables",
    "/api/2.1/synced-tables",
    "/api/2.0/delta-sync-tables",
    "/api/2.1/unity-catalog/synced-tables",
]:
    r = requests.get(f"{host}{path}", headers=headers)
    print(f"{path}: {r.status_code} {r.text[:200]}")

# COMMAND ----------
# The task says use "online/low-latency access" path
# In Databricks, this can be done via:
# 1. Synced Tables (replacement for Online Tables)
# 2. Feature Serving Endpoints
# 3. Delta Sharing or other mechanisms

# Let's try the Feature Engineering API which might support synced tables
print("=== Feature Engineering API ===")
for path in [
    "/api/2.0/feature-engineering/feature-tables",
    "/api/2.0/feature-store/feature-tables",
    "/api/2.1/feature-engineering",
    "/api/2.0/feature-engineering/online-feature-tables",
]:
    r = requests.get(f"{host}{path}", headers=headers)
    print(f"{path}: {r.status_code} {r.text[:300]}")

# COMMAND ----------
# Look at the Databricks REST API openapi spec to find synced tables
print("=== OpenAPI endpoints check ===")
for path in [
    "/api/2.0/openapi.json",
    "/api/2.0/synced_tables",
    "/api/2.0/tables/synced",
    "/api/2.1/online-tables",
]:
    r = requests.get(f"{host}{path}", headers=headers)
    print(f"{path}: {r.status_code}")
    if r.status_code == 200:
        print(r.text[:500])

# COMMAND ----------
# Check if we can list or create synced tables via UC REST API
print("=== Unity Catalog Synced Tables ===")
# Try creating a synced table
synced_payload = {
    "name": f"{schema}.{table_name}_synced",
    "source_table": full_table_name,
    "pipeline_type": "TRIGGERED"
}
for path in ["/api/2.0/synced-tables", "/api/2.1/synced-tables"]:
    r = requests.post(f"{host}{path}", headers=headers, json=synced_payload)
    print(f"POST {path}: {r.status_code} {r.text[:500]}")

# COMMAND ----------
# Try to look at the feature serving endpoints API
print("=== Feature Serving Endpoints ===")
r = requests.get(f"{host}/api/2.0/serving-endpoints", headers=headers)
print(f"List endpoints: {r.status_code} {r.text[:500]}")

# Try creating a feature serving endpoint directly using the online table
# (even though online table creation failed, maybe there's a direct query method)

# COMMAND ----------
# Try reading from the Delta table directly via SQL and see if we can serve it
print("=== Direct SQL Query ===")
df = spark.sql(f"SELECT * FROM {full_table_name}")
df.show()
results = {}
for row in df.collect():
    results[row.account_id] = [row.f1, row.f2, row.f3, row.f4]
print(f"Got {len(results)} rows from offline store")

# COMMAND ----------
# Given online tables are deprecated, let's try the Synced Tables via DLT or other mechanism
# Check what pipelines can do
print("=== DLT / Pipeline approach ===")
for path in [
    "/api/2.0/pipelines",
    "/api/2.1/pipelines",
]:
    r = requests.get(f"{host}{path}", headers=headers)
    print(f"{path}: {r.status_code} {r.text[:300]}")

# COMMAND ----------
# Final: output what we know about synced tables in this workspace
debug_output = {
    "online_table_deprecated": True,
    "recommendation": "Use Synced Tables",
    "rows_in_offline_table": len(results)
}
print(json.dumps(debug_output, indent=2))
dbutils.notebook.exit(json.dumps(debug_output))
