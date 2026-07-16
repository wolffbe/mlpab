# Databricks notebook source
# COMMAND ----------
import requests, json

host = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

lines = []

# Check UC tables API for listing all table types
r = requests.get(f"https://{host}/api/2.1/unity-catalog/tables?catalog_name=workspace&schema_name=mlpab6ef9cb", headers=headers)
lines.append(f"UC tables list status: {r.status_code}")
if r.status_code == 200:
    data = r.json()
    for t in data.get('tables', []):
        lines.append(f"  table: {t.get('name')} type={t.get('table_type')} delta={t.get('delta_runtime_properties_kvpairs')}")

# Check what table_type options exist
r2 = requests.get(f"https://{host}/api/2.1/unity-catalog/tables/workspace.mlpab6ef9cb.scores4f5893", headers=headers)
if r2.status_code == 200:
    data2 = r2.json()
    lines.append(f"\nTable details: name={data2.get('name')}, type={data2.get('table_type')}, delta_props={data2.get('delta_runtime_properties_kvpairs')}")
    lines.append(f"Properties: {data2.get('properties')}")

# Let's check if there's something like a catalog/synced table PATCH to add online capability
# Or check if there's a feature store table update API
r3 = requests.get(f"https://{host}/api/2.0/unity-catalog/tables/workspace.mlpab6ef9cb.scores4f5893", headers=headers)
if r3.status_code == 200:
    d3 = r3.json()
    lines.append(f"\nv2.0 properties: {d3.get('properties')}")

dbutils.notebook.exit("\n".join(lines))
