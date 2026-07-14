# Databricks notebook source
import json
import time

ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
token = ctx.apiToken().get()
host = ctx.apiUrl().get()

schema = "workspace.mlpabcef85c"
table_name = "profilesaa70e4"
full_table_name = f"{schema}.{table_name}"
prefix = "mlpabcef85c"

# COMMAND ----------
# Check what's available in online_tables
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
print("online_tables methods:", [m for m in dir(w.online_tables) if not m.startswith('_')])

# COMMAND ----------
# Check what classes are in catalog service
import databricks.sdk.service.catalog as cat_module
online_table_classes = [x for x in dir(cat_module) if 'online' in x.lower() or 'Online' in x]
print("Online table classes:", online_table_classes)

# COMMAND ----------
# Create the online table using available SDK classes
from databricks.sdk.service import catalog

# Try to get what we need
print("OnlineTable:", hasattr(catalog, 'OnlineTable'))
print("OnlineTableSpec:", hasattr(catalog, 'OnlineTableSpec'))
print("OnlineTableSpecTriggeredSchedulingPolicy:", hasattr(catalog, 'OnlineTableSpecTriggeredSchedulingPolicy'))

# Find triggered policy class
triggered_classes = [x for x in dir(catalog) if 'Triggered' in x or 'triggered' in x]
print("Triggered classes:", triggered_classes)

# COMMAND ----------
# Create the online table via SDK
from databricks.sdk.service.catalog import OnlineTable, OnlineTableSpec

# Find the correct triggered class
all_policy_classes = [x for x in dir(catalog) if 'Policy' in x or 'Schedule' in x or 'Triggered' in x]
print("Policy/Schedule classes:", all_policy_classes)

# Build spec with available classes
spec_args = {
    "source_table_full_name": full_table_name,
    "primary_key_columns": ["account_id"],
}

# Try adding triggered policy
if hasattr(catalog, 'OnlineTableSpecTriggeredSchedulingPolicy'):
    spec_args["run_triggered"] = catalog.OnlineTableSpecTriggeredSchedulingPolicy()
elif hasattr(catalog, 'RunTriggered'):
    spec_args["run_triggered"] = catalog.RunTriggered()

print(f"Creating online table with spec args: {spec_args}")

try:
    spec = OnlineTableSpec(**spec_args)
    print(f"Spec created: {spec}")

    result = w.online_tables.create(
        name=full_table_name,
        spec=spec
    )
    print(f"Create result: {result}")
except Exception as e:
    print(f"Create failed: {type(e).__name__}: {e}")
    # Try with dict spec
    try:
        import requests
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        payload = {
            "name": full_table_name,
            "spec": {
                "source_table_full_name": full_table_name,
                "primary_key_columns": ["account_id"],
                "run_triggered": {}
            }
        }
        r = requests.post(f"{host}/api/2.0/online-tables", headers=headers, json=payload)
        print(f"REST create: {r.status_code} {r.text[:500]}")
    except Exception as e2:
        print(f"REST create also failed: {e2}")

# COMMAND ----------
# Check if the online table was created
try:
    ot = w.online_tables.get(name=full_table_name)
    print(f"Online table: {ot}")
except Exception as e:
    print(f"Get failed: {e}")

# COMMAND ----------
# Get the SDK version
import databricks.sdk
print(f"SDK version: {databricks.sdk.__version__}")

# Inspect the online_tables create method
import inspect
print("Create signature:", inspect.signature(w.online_tables.create))

# COMMAND ----------
dbutils.notebook.exit("check complete")
