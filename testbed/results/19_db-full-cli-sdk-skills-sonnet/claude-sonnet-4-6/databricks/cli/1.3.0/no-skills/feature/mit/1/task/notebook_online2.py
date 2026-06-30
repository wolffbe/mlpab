# Databricks notebook source
# COMMAND ----------
# Check SDK version and available classes
import databricks.sdk
print("SDK version:", databricks.sdk.__version__)

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog as sdk_catalog
print("Online table classes:", [c for c in dir(sdk_catalog) if 'Online' in c or 'Synced' in c or 'Triggered' in c])

# COMMAND ----------
# Try to create an online table using the available SDK classes
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

# Check what's available in online_tables
print("online_tables methods:", [m for m in dir(w.online_tables) if not m.startswith('_')])

# Try creating online table with available spec
try:
    from databricks.sdk.service.catalog import OnlineTableSpec

    online_table = w.online_tables.create(
        name="workspace.mlpabf1452c.featuresb1ea93_online",
        spec=OnlineTableSpec(
            source_table_full_name="workspace.mlpabf1452c.featuresb1ea93",
            primary_key_columns=["row_id"],
            timeseries_key="event_time",
            run_triggered={}
        )
    )
    print(f"Online table creation result: {online_table}")
except Exception as e:
    print(f"Online table creation failed: {type(e).__name__}: {e}")

# COMMAND ----------
# Check for synced tables in SDK
try:
    print("Checking synced_tables:", hasattr(w, 'synced_tables'))
    if hasattr(w, 'synced_tables'):
        print("synced_tables methods:", [m for m in dir(w.synced_tables) if not m.startswith('_')])
except Exception as e:
    print(f"Error checking synced_tables: {e}")
