# Databricks notebook source
# Find the correct field names for synced_table
import requests
host = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

results = []

# Check get-synced-table API
r = requests.get(f"https://{host}/api/2.0/postgres/synced_tables", headers=headers,
    params={"synced_table_id": "workspace.mlpab0442b8.accountse81ff1"})
results.append(f"GET synced_tables: {r.status_code} {r.text[:300]}")

# Check what synced tables exist for other projects
for project_id in ["mlpab0442b8-lakebase", "mlpabefbb2e-feat", "mlpab0442b8-online-store"]:
    r2 = requests.get(f"https://{host}/api/2.0/postgres/synced_tables", headers=headers,
        params={"parent": f"projects/{project_id}/branches/production"})
    results.append(f"GET synced_tables for {project_id}: {r2.status_code} {r2.text[:300]}")

# Try a proto3 approach - send a numeric enum value for a subfield
r3 = requests.post(f"https://{host}/api/2.0/postgres/synced_tables", headers=headers,
    params={"synced_table_id": "mlpab0442b8db.mlpab0442b8.accountse81ff1"},
    json={"synced_table": {"status": 1}})
results.append(f"Status=1: {r3.status_code} {r3.text[:200]}")

# Try with type annotation
r4 = requests.post(f"https://{host}/api/2.0/postgres/synced_tables", headers=headers,
    params={"synced_table_id": "mlpab0442b8db.mlpab0442b8.accountse81ff1"},
    json={"synced_table": {"name": "test"}})
results.append(f"name=test: {r4.status_code} {r4.text[:200]}")

# Try looking at other synced tables to understand schema
r5 = requests.get(f"https://{host}/api/2.0/postgres/synced_tables", headers=headers,
    params={"parent": "projects/mlpab0442b8-lakebase/branches/production"})
results.append(f"Existing synced tables: {r5.status_code} {r5.text[:500]}")

spark.createDataFrame([(r,) for r in results], ["result"]).write.mode("overwrite").saveAsTable("workspace.mlpab0442b8.synced_fields_discovery")
