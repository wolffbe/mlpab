import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import SyncedTableSpec, TriggerType

w = WorkspaceClient()

try:
    synced_table = w.synced_tables.create(
        name="workspace.mlpabbb38f1.derivedd05474_online",
        spec=SyncedTableSpec(
            source_table_full_name="workspace.mlpabbb38f1.derivedd05474",
            primary_key_columns=["row_id"],
            run_triggered={}
        )
    )
    print(f"Created synced table: {synced_table}")
except Exception as e:
    print(f"SyncedTables error: {e}")
    try:
        online_table = w.online_tables.create(
            name="workspace.mlpabbb38f1.derivedd05474_online",
            spec={
                "source_table_full_name": "workspace.mlpabbb38f1.derivedd05474",
                "primary_key_columns": ["row_id"],
                "run_triggered": {}
            }
        )
        print(f"Created online table: {online_table}")
    except Exception as e2:
        print(f"OnlineTables error: {e2}")
