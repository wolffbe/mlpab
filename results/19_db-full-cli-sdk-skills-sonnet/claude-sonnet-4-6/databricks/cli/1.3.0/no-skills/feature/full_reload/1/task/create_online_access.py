# Databricks notebook source

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.database import (
    SyncedDatabaseTable, SyncedTableSpec, SyncedTableSchedulingPolicy
)

w = WorkspaceClient()
SOURCE_TABLE = "workspace.mlpab2c4304.customersc31b07"
ONLINE_TABLE = "workspace.mlpab2c4304.customersc31b07_online"
DB_INSTANCE = "mlpab2c4304-lakebase"
LOGICAL_DB = "customerdb"

output_lines = []

# COMMAND ----------

# Enable Change Data Feed on the source table
spark.sql(f"ALTER TABLE {SOURCE_TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
output_lines.append("CDF enabled on source table.")

# COMMAND ----------

# Create synced table for online/real-time access
try:
    spec = SyncedTableSpec(
        source_table_full_name=SOURCE_TABLE,
        primary_key_columns=["row_id"],
        timeseries_key="updated_at",
        create_database_objects_if_missing=True,
        scheduling_policy=SyncedTableSchedulingPolicy.TRIGGERED
    )
    synced_table = SyncedDatabaseTable(
        name=ONLINE_TABLE,
        database_instance_name=DB_INSTANCE,
        logical_database_name=LOGICAL_DB,
        spec=spec
    )
    result = w.database.create_synced_database_table(synced_table=synced_table)
    output_lines.append("SUCCESS: " + repr(result))
except Exception as e:
    output_lines.append("Error: " + str(e))
    import traceback
    output_lines.append(traceback.format_exc())

dbutils.notebook.exit("\n".join(output_lines))
