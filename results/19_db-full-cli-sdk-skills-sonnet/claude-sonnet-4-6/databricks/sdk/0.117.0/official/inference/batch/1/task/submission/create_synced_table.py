"""
Create a Lakebase database instance and a synced table for online/real-time access.
"""
import os
import time
from datetime import timedelta

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound, BadRequest
from databricks.sdk.service.database import (
    DatabaseInstance,
    SyncedDatabaseTable,
    SyncedTableSpec,
    SyncedTableSchedulingPolicy,
    NewPipelineSpec,
)

SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]    # workspace.mlpab69a58e
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]     # mlpab69a58e
FULL_TABLE_NAME = f"{SCHEMA}.scores4f5893"
DB_INSTANCE_NAME = f"{PREFIX}-db"

w = WorkspaceClient()
print(f"Connected to Databricks: {w.config.host}")
print(f"Schema: {SCHEMA}")
print(f"Full table name: {FULL_TABLE_NAME}")
print(f"DB instance name: {DB_INSTANCE_NAME}")

# Step 1: Create Lakebase database instance
print(f"\nCreating Lakebase database instance '{DB_INSTANCE_NAME}'...")
try:
    existing = w.database.get_database_instance(name=DB_INSTANCE_NAME)
    print(f"Instance already exists: {existing.name} (state={existing.state})")
    db_instance = existing
except NotFound:
    print(f"Instance not found, creating...")
    waiter = w.database.create_database_instance(
        database_instance=DatabaseInstance(
            name=DB_INSTANCE_NAME,
            capacity="CU_1",
        )
    )
    print(f"Waiting for database instance to be available...")
    db_instance = waiter.result(timeout=timedelta(seconds=600))
    print(f"Database instance ready: {db_instance.name}")

print(f"Database instance: {db_instance.name}, state={db_instance.state}")

# Step 2: Create the synced table
catalog_name, schema_name = SCHEMA.split(".", 1)
logical_db_name = schema_name  # Use schema name as the logical database name

print(f"\nCreating synced database table for {FULL_TABLE_NAME}...")
try:
    existing_sync = w.database.get_synced_database_table(name=FULL_TABLE_NAME)
    print(f"Synced table already exists: {existing_sync.name}")
except (NotFound, BadRequest):
    print(f"Creating synced table...")
    synced_table = w.database.create_synced_database_table(
        synced_table=SyncedDatabaseTable(
            name=FULL_TABLE_NAME,
            database_instance_name=DB_INSTANCE_NAME,
            logical_database_name=logical_db_name,
            spec=SyncedTableSpec(
                source_table_full_name=FULL_TABLE_NAME,
                primary_key_columns=["account_id"],
                scheduling_policy=SyncedTableSchedulingPolicy.TRIGGERED,
                create_database_objects_if_missing=True,
                new_pipeline_spec=NewPipelineSpec(
                    storage_catalog=catalog_name,
                    storage_schema=schema_name,
                ),
            ),
        )
    )
    print(f"Synced table creation initiated: {synced_table.name}")
    print(f"State: {synced_table.unity_catalog_provisioning_state}")

# Wait for sync to complete
print("\nWaiting for synced table to be ready...")
max_wait = 600
start = time.time()
while time.time() - start < max_wait:
    try:
        st = w.database.get_synced_database_table(name=FULL_TABLE_NAME)
        state = str(st.unity_catalog_provisioning_state)
        sync_status = st.data_synchronization_status
        print(f"Unity Catalog state: {state}, Sync status: {sync_status}")
        if "ACTIVE" in state.upper() or "SUCCEEDED" in state.upper():
            print("Synced table is ready!")
            break
        elif "FAILED" in state.upper() or "ERROR" in state.upper():
            print(f"Synced table failed: {st}")
            break
        time.sleep(20)
    except Exception as e:
        print(f"Status check error: {e}")
        time.sleep(20)

# Final status
try:
    st = w.database.get_synced_database_table(name=FULL_TABLE_NAME)
    print(f"\nFinal synced table status:")
    print(f"  Name: {st.name}")
    print(f"  Database instance: {st.effective_database_instance_name}")
    print(f"  Logical DB: {st.effective_logical_database_name}")
    print(f"  UC state: {st.unity_catalog_provisioning_state}")
    print(f"  Sync status: {st.data_synchronization_status}")
except Exception as e:
    print(f"Could not get final status: {e}")

print("\nDone! Online/real-time access enabled via Lakebase Synced Table.")
