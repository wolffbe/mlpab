# Databricks notebook source

ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
token = ctx.apiToken().get()
host = ctx.apiUrl().get()

import requests
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

# COMMAND ----------

# Try different API paths for synced tables
paths_to_try = [
    "/api/2.0/synced-tables",
    "/api/2.1/synced-tables",
    "/api/2.0/preview/synced-tables",
    "/api/2.0/catalog/synced-tables",
    "/api/2.0/feature-store/synced-tables",
]

for path in paths_to_try:
    resp = requests.get(f"{host}{path}", headers=headers)
    print(f"GET {path}: {resp.status_code}")

# COMMAND ----------

# Try creating via the DataBricks SDK
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import SyncedTableSpec

w = WorkspaceClient()
print(f"SDK client created: {w}")

# List available methods
print(dir(w))

# COMMAND ----------

# Try w.synced_tables or similar
attrs = [a for a in dir(w) if 'sync' in a.lower() or 'online' in a.lower() or 'feature' in a.lower()]
print("Relevant attributes:", attrs)

# COMMAND ----------

# Check if synced_tables is available
if hasattr(w, 'synced_tables'):
    print("synced_tables available!")
    print(dir(w.synced_tables))
elif hasattr(w, 'online_tables'):
    print("online_tables available!")
    print(dir(w.online_tables))
else:
    print("Neither synced_tables nor online_tables available in SDK")

# COMMAND ----------

# Try to create synced table via SDK
try:
    from databricks.sdk.service.catalog import SyncedTable, SyncedTableSpec, TriggeredSchedule
    spec = SyncedTableSpec(
        source_table_full_name="workspace.mlpabd7768b.airqpredfdfb59",
        primary_key_columns=["date"],
        run_triggered=TriggeredSchedule()
    )
    table = SyncedTable(
        name="workspace.mlpabd7768b.airqpredfdfb59_synced",
        spec=spec
    )
    result = w.synced_tables.create_and_wait(table)
    print(f"Created synced table: {result}")
except Exception as e:
    print(f"Error creating synced table: {e}")

# COMMAND ----------

dbutils.notebook.exit("done")
