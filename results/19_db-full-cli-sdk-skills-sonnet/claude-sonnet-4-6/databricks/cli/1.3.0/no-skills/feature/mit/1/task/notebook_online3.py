# Databricks notebook source
# COMMAND ----------
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog as sdk_catalog

# Check available online/synced table classes
online_classes = [c for c in dir(sdk_catalog) if 'Online' in c or 'Synced' in c or 'Triggered' in c]
print("Available classes:", online_classes)

w = WorkspaceClient()
print("Has online_tables:", hasattr(w, 'online_tables'))
print("Has synced_tables:", hasattr(w, 'synced_tables'))

# COMMAND ----------
# Try creating online table with just OnlineTableSpec
try:
    from databricks.sdk.service.catalog import OnlineTableSpec
    print("OnlineTableSpec available")
    print("OnlineTableSpec fields:", [f for f in dir(OnlineTableSpec) if not f.startswith('_')])
except ImportError as e:
    print(f"OnlineTableSpec not available: {e}")

# COMMAND ----------
# Try creating online table
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import OnlineTableSpec

w = WorkspaceClient()

try:
    # Try with run_triggered as an empty dict-like object
    online_table = w.online_tables.create(
        name="workspace.mlpabf1452c.featuresb1ea93_online",
        spec=OnlineTableSpec(
            source_table_full_name="workspace.mlpabf1452c.featuresb1ea93",
            primary_key_columns=["row_id"],
            timeseries_key="event_time"
        )
    )
    print(f"Online table created: {online_table}")
except Exception as e:
    print(f"Failed: {type(e).__name__}: {e}")
