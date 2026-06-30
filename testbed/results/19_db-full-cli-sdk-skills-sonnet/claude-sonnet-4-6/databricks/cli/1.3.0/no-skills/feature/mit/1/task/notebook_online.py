# Databricks notebook source
# COMMAND ----------
# Try to create a synced table for online access

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import OnlineTableSpec, OnlineTableSpecTriggeredSchedulingPolicy

w = WorkspaceClient()

# Try to create an online table via the SDK
try:
    online_table = w.online_tables.create(
        name="workspace.mlpabf1452c.featuresb1ea93_online",
        spec=OnlineTableSpec(
            source_table_full_name="workspace.mlpabf1452c.featuresb1ea93",
            primary_key_columns=["row_id"],
            timeseries_key="event_time",
            run_triggered=OnlineTableSpecTriggeredSchedulingPolicy()
        )
    )
    print(f"Online table created: {online_table}")
except Exception as e:
    print(f"Online table creation failed: {e}")

# COMMAND ----------
# Try synced tables approach via SQL
try:
    result = spark.sql("""
        CREATE OR REPLACE TABLE workspace.mlpabf1452c.featuresb1ea93_online
        USING synced_table
        AS SELECT * FROM workspace.mlpabf1452c.featuresb1ea93
    """)
    print("Synced table created via SQL")
except Exception as e:
    print(f"Synced table SQL failed: {e}")

# COMMAND ----------
# Check available methods in SDK
import databricks
print(dir(w))
# Check if there's a synced tables client
try:
    print("synced_tables:", w.synced_tables)
except Exception as e:
    print(f"No synced_tables in SDK: {e}")
