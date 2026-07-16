# Databricks notebook source
import databricks.sdk.service.database as db_svc
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
results = []

# Create SyncedTableSpec with source table
spec = db_svc.SyncedTableSpec(
    source_table_full_name="workspace.mlpab0442b8.accountse81ff1",
    primary_key_columns=["row_id", "updated_at"],
    timeseries_key="updated_at",
    create_database_objects_if_missing=True,
    scheduling_policy=db_svc.SyncedTableSchedulingPolicy.TRIGGERED
)
results.append(f"Spec: {spec.as_dict()}")

# Create SyncedDatabaseTable
synced_table = db_svc.SyncedDatabaseTable(
    name="mlpab0442b8db.mlpab0442b8.accountse81ff1",
    spec=spec
)
results.append(f"SyncedDatabaseTable: {synced_table.as_dict()}")

# Create it using the SDK
try:
    result = w.database.create_synced_database_table(synced_table=synced_table)
    results.append(f"Success! Result: {result}")
except Exception as e:
    results.append(f"Error: {type(e).__name__}: {e}")

spark.createDataFrame([(r,) for r in results], ["result"]).write.mode("overwrite").saveAsTable("workspace.mlpab0442b8.synced_db_table_output")
