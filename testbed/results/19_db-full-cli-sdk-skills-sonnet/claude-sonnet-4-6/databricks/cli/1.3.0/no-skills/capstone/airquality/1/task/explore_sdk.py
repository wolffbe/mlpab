# Databricks notebook source

# COMMAND ----------

import databricks.sdk
print(f"SDK version: {databricks.sdk.__version__}")

from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

# Find online/synced table related attributes
attrs = [a for a in dir(w) if any(x in a.lower() for x in ['sync', 'online', 'feature'])]
print("Relevant SDK attributes:", attrs)

# COMMAND ----------

# Check what classes are in catalog service
import databricks.sdk.service.catalog as cat
catalog_attrs = [a for a in dir(cat) if any(x in a.lower() for x in ['sync', 'online', 'feature'])]
print("Catalog service classes:", catalog_attrs)

# COMMAND ----------

# Try w.online_tables
if hasattr(w, 'online_tables'):
    print("online_tables:", dir(w.online_tables))

# COMMAND ----------

# Try to list online tables via raw API
ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
token = ctx.apiToken().get()
host = ctx.apiUrl().get()

import requests
headers = {"Authorization": f"Bearer {token}"}

# Check serving endpoints
resp = requests.get(f"{host}/api/2.0/serving-endpoints", headers=headers)
print(f"Serving endpoints: {resp.status_code}")
if resp.ok:
    print(resp.json())

# COMMAND ----------

# Try creating an online table with the SDK if available
if hasattr(w, 'online_tables'):
    try:
        from databricks.sdk.service import catalog as cat_svc
        # Try creating directly
        result = w.online_tables.create(
            name="workspace.mlpabd7768b.airqpredfdfb59_online",
            spec=cat_svc.OnlineTableSpec(
                source_table_full_name="workspace.mlpabd7768b.airqpredfdfb59",
                primary_key_columns=["date"],
                run_triggered=cat_svc.OnlineTableSpecTriggeredSchedulingPolicy()
            )
        )
        print(f"Created: {result}")
    except Exception as e:
        print(f"Failed: {e}")
else:
    print("No online_tables in SDK")

# COMMAND ----------

dbutils.notebook.exit("sdk_exploration_done")
