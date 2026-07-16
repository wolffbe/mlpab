"""Create Lakebase database instance and synced table for low-latency lookup."""
import os
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.database import (
    DatabaseInstance,
    SyncedDatabaseTable,
    SyncedTableSpec,
    SyncedTableSchedulingPolicy,
)

w = WorkspaceClient()
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]
SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]
CATALOG = SCHEMA.split(".")[0]
DB = SCHEMA.split(".")[1]

PRED_NAME = "airqpredf4aae3"
source_table = f"{CATALOG}.{DB}.{PRED_NAME}"
DB_INSTANCE_NAME = f"{PREFIX}-lakebase"

print(f"Source table: {source_table}")
print(f"DB instance: {DB_INSTANCE_NAME}")
print()

# Step 1: Check for existing database instances
print("--- Checking existing database instances ---")
try:
    existing = list(w.database.list_database_instances())
    print(f"Existing instances: {[i.name for i in existing]}")
    # Check if our instance already exists
    our_instance = next((i for i in existing if i.name == DB_INSTANCE_NAME), None)
except Exception as e:
    print(f"List instances failed: {e}")
    existing = []
    our_instance = None

# Step 2: Create database instance if needed
if our_instance is None:
    print(f"\n--- Creating database instance: {DB_INSTANCE_NAME} ---")
    try:
        instance = w.database.create_database_instance(
            DatabaseInstance(
                name=DB_INSTANCE_NAME,
                capacity="CU_1",
            )
        )
        print(f"Instance created: {instance.name} state={instance.state}")
        # Wait for it to be available
        print("Waiting for instance to be available...")
        deadline = time.time() + 600  # 10 min max
        while time.time() < deadline:
            inst = w.database.get_database_instance(name=DB_INSTANCE_NAME)
            state = inst.state
            print(f"  State: {state}")
            if str(state) == "DatabaseInstanceState.AVAILABLE":
                print("Instance is AVAILABLE!")
                break
            if str(state) in ("DatabaseInstanceState.FAILED", "DatabaseInstanceState.DELETED"):
                print(f"Instance failed: {state}")
                break
            time.sleep(15)
        our_instance = inst
    except Exception as e:
        print(f"Create instance failed: {e}")
        # Try to get it anyway
        try:
            our_instance = w.database.get_database_instance(name=DB_INSTANCE_NAME)
            print(f"Instance exists: {our_instance.state}")
        except Exception as e2:
            print(f"Get instance failed: {e2}")
else:
    print(f"Using existing instance: {our_instance.name} state={our_instance.state}")

print()

# Step 3: Check for existing synced table
print("--- Checking for existing synced table ---")
try:
    existing_st = w.database.get_synced_database_table(name=source_table)
    print(f"Synced table exists: {existing_st.name} state={existing_st.unity_catalog_provisioning_state}")
    print(f"  Sync status: {existing_st.data_synchronization_status}")
    print("SUCCESS: Synced table already exists!")
    exit(0)
except Exception as e:
    print(f"Get synced table: {e}")

# Step 4: Create synced database table
print(f"\n--- Creating synced database table: {source_table} ---")
try:
    synced_table = w.database.create_synced_database_table(
        SyncedDatabaseTable(
            name=source_table,
            database_instance_name=DB_INSTANCE_NAME,
            spec=SyncedTableSpec(
                source_table_full_name=source_table,
                primary_key_columns=["date"],
                scheduling_policy=SyncedTableSchedulingPolicy.TRIGGERED,
                create_database_objects_if_missing=True,
            ),
        )
    )
    print(f"Synced table created: {synced_table.name}")
    print(f"  UC state: {synced_table.unity_catalog_provisioning_state}")
    print(f"  Sync status: {synced_table.data_synchronization_status}")
    print(f"  DB instance: {synced_table.effective_database_instance_name}")
    print(f"  Logical DB: {synced_table.effective_logical_database_name}")

    # Wait for initial sync
    print("\nWaiting for sync to complete...")
    deadline = time.time() + 600  # 10 min max
    while time.time() < deadline:
        st = w.database.get_synced_database_table(name=source_table)
        uc_state = st.unity_catalog_provisioning_state
        sync_status = st.data_synchronization_status
        print(f"  UC state: {uc_state}, sync: {sync_status}")
        if sync_status and hasattr(sync_status, "state"):
            ss = str(sync_status.state)
            if "SYNCED" in ss or "ACTIVE" in ss or "DONE" in ss:
                print("Synced table is synced!")
                break
            if "FAILED" in ss:
                print(f"Sync failed: {sync_status}")
                break
        if str(uc_state) == "ProvisioningInfoState.ACTIVE":
            print("UC state is ACTIVE!")
            break
        time.sleep(15)

    print(f"\nFinal state: {st.unity_catalog_provisioning_state}")
    print("SUCCESS: Synced table created and syncing")

except Exception as e:
    print(f"Create synced table failed: {type(e).__name__}: {e}")
    # Try to list synced tables to see what exists
    try:
        tables = list(w.database.list_synced_database_tables())
        print(f"Existing synced tables: {[t.name for t in tables]}")
    except Exception as e2:
        print(f"List synced tables: {e2}")
