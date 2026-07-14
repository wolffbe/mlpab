# Databricks notebook source
# COMMAND ----------
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import SyncedTableSpec, SyncedTableRunTriggered

w = WorkspaceClient()

# Create a synced table for low-latency lookup
try:
    result = w.synced_tables.create(
        name="workspace.mlpab6ef9cb.scores4f5893_synced",
        spec=SyncedTableSpec(
            source_table_full_name="workspace.mlpab6ef9cb.scores4f5893",
            primary_key_columns=["account_id"],
            run_triggered=SyncedTableRunTriggered()
        )
    )
    print(f"Synced table created: {result}")
except Exception as e:
    print(f"Error creating synced table: {e}")
    import traceback
    traceback.print_exc()
