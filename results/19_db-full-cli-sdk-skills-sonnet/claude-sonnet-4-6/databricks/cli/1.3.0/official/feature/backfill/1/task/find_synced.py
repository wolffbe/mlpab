# Databricks notebook source
# Find Synced Tables in SDK

# COMMAND ----------
from databricks.sdk import WorkspaceClient
import inspect

w = WorkspaceClient()
results = []

# List all services that might have synced table methods
all_services = [attr for attr in dir(w) if not attr.startswith('_')]
synced_related = [s for s in all_services if 'synced' in s.lower() or 'online' in s.lower() or 'serving' in s.lower()]
results.append(f"Synced/online/serving services: {synced_related}")

# Check if there are any synced table classes
try:
    from databricks.sdk.service import catalog as cat_service
    synced_classes = [c for c in dir(cat_service) if 'synced' in c.lower() or 'online' in c.lower()]
    results.append(f"Catalog service synced/online classes: {synced_classes}")
except Exception as e:
    results.append(f"catalog service: {e}")

# Check what's in databricks.sdk.service
from databricks.sdk import service
all_modules = [m for m in dir(service) if not m.startswith('_')]
results.append(f"SDK service modules: {all_modules}")

print('\n'.join(results))
spark.createDataFrame([(r,) for r in results], ["result"]).write.mode("overwrite").saveAsTable("workspace.mlpab0442b8.synced_discovery")

# COMMAND ----------
# Try to use REST API to create synced table within notebook
import requests

host = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

results2 = []

# Check if online-tables GET works (might have list endpoint)
r = requests.get(f"https://{host}/api/2.0/online-tables", headers=headers)
results2.append(f"GET /api/2.0/online-tables: {r.status_code} {r.text[:200]}")

# Try POST to online-tables (the REST API might allow it even if CLI doesn't)
payload = {
    "name": "workspace.mlpab0442b8.accountse81ff1_online",
    "spec": {
        "source_table_full_name": "workspace.mlpab0442b8.accountse81ff1",
        "primary_key_columns": ["row_id", "updated_at"],
        "run_triggered": {}
    }
}
r2 = requests.post(f"https://{host}/api/2.0/online-tables", headers=headers, json=payload)
results2.append(f"POST /api/2.0/online-tables: {r2.status_code} {r2.text[:300]}")

# Try synced-tables
r3 = requests.post(f"https://{host}/api/2.0/synced-tables", headers=headers, json=payload)
results2.append(f"POST /api/2.0/synced-tables: {r3.status_code} {r3.text[:200]}")

print('\n'.join(results2))
spark.createDataFrame([(r,) for r in results2], ["result"]).write.mode("append").saveAsTable("workspace.mlpab0442b8.synced_discovery")
