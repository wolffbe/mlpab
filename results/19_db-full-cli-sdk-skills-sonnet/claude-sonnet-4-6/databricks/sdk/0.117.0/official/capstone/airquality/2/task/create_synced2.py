"""Create Lakebase synced table - fixed version."""
import os
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.database import (
    DatabaseInstance,
    DatabaseInstanceState,
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
LOGICAL_DB_NAME = DB  # Use the same name as our UC schema

print(f"Source table: {source_table}")
print(f"DB instance: {DB_INSTANCE_NAME}")
print(f"Logical DB name: {LOGICAL_DB_NAME}")
print()

# Step 1: Get or create database instance
print("--- Database Instance ---")
try:
    inst = w.database.get_database_instance(name=DB_INSTANCE_NAME)
    print(f"Found instance: {inst.name} state={inst.state}")
except Exception as e:
    print(f"Instance not found, creating: {e}")
    try:
        # Just create it - ignore deserialization issues
        import requests
        host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
        if not host.startswith("http"):
            host = "https://" + host
        token = os.environ.get("DATABRICKS_TOKEN", "")
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        r = requests.post(
            f"{host}/api/2.0/database/instances",
            headers=headers,
            json={"name": DB_INSTANCE_NAME, "capacity": "CU_1"}
        )
        print(f"Create via REST: {r.status_code} {r.text[:300]}")
    except Exception as e2:
        print(f"REST create failed: {e2}")

# Wait for instance to be AVAILABLE
print("Waiting for database instance to be AVAILABLE...")
deadline = time.time() + 600
while time.time() < deadline:
    try:
        inst = w.database.get_database_instance(name=DB_INSTANCE_NAME)
        state = inst.state
        print(f"  Instance state: {state}")
        if state == DatabaseInstanceState.AVAILABLE:
            print("Instance is AVAILABLE!")
            break
        if state in (DatabaseInstanceState.DELETED, DatabaseInstanceState.FAILED):
            print(f"Instance in terminal state: {state}")
            break
    except Exception as e:
        print(f"  Get instance: {e}")
    time.sleep(15)

print()

# Step 2: Check for existing synced table
print("--- Synced Table ---")
try:
    st = w.database.get_synced_database_table(name=source_table)
    print(f"Synced table exists: {st.name}")
    print(f"  UC state: {st.unity_catalog_provisioning_state}")
    print(f"  DB instance: {st.effective_database_instance_name}")
    print(f"  Logical DB: {st.effective_logical_database_name}")
    print(f"  Sync status: {st.data_synchronization_status}")
    print("SUCCESS: Synced table already exists!")
    exit(0)
except Exception as e:
    print(f"Synced table not found: {e}")

# Step 3: Create synced table with logical_database_name
print(f"\nCreating synced table: {source_table}")
print(f"  logical_database_name: {LOGICAL_DB_NAME}")

try:
    synced_table = w.database.create_synced_database_table(
        SyncedDatabaseTable(
            name=source_table,
            database_instance_name=DB_INSTANCE_NAME,
            logical_database_name=LOGICAL_DB_NAME,
            spec=SyncedTableSpec(
                source_table_full_name=source_table,
                primary_key_columns=["date"],
                scheduling_policy=SyncedTableSchedulingPolicy.TRIGGERED,
                create_database_objects_if_missing=True,
            ),
        )
    )
    print(f"Created! Name: {synced_table.name}")
    print(f"  UC state: {synced_table.unity_catalog_provisioning_state}")
    print(f"  DB instance: {synced_table.effective_database_instance_name}")
    print(f"  Logical DB: {synced_table.effective_logical_database_name}")
    print(f"  Sync status: {synced_table.data_synchronization_status}")

    # Wait for sync
    print("\nWaiting for initial sync...")
    deadline = time.time() + 600
    while time.time() < deadline:
        try:
            st = w.database.get_synced_database_table(name=source_table)
            uc_state = str(st.unity_catalog_provisioning_state)
            sync = st.data_synchronization_status
            sync_str = str(sync) if sync else "None"
            print(f"  UC={uc_state} sync={sync_str[:100]}")
            if "ACTIVE" in uc_state:
                print("Synced table is ACTIVE!")
                break
            if "FAILED" in uc_state:
                print(f"Synced table FAILED")
                break
        except Exception as e:
            print(f"  Get error: {e}")
        time.sleep(20)

    print("\nFinal state:")
    try:
        st = w.database.get_synced_database_table(name=source_table)
        print(f"  Name: {st.name}")
        print(f"  UC state: {st.unity_catalog_provisioning_state}")
        print(f"  Sync: {st.data_synchronization_status}")
        print("SUCCESS: Synced table created for low-latency lookup!")
    except Exception as e:
        print(f"  Final get failed: {e}")

except Exception as e:
    print(f"Create synced table failed: {type(e).__name__}: {e}")
