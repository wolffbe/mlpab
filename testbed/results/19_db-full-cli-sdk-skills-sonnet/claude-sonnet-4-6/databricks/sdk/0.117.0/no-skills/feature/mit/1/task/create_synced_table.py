import os
import time
from databricks.sdk import WorkspaceClient
import databricks.sdk.service.database as dbsvc
from databricks.sdk.errors import NotFound

w = WorkspaceClient()

schema_full = os.environ["MLPAB_DATABRICKS_SCHEMA"]
prefix = os.environ["MLPAB_DATABRICKS_PREFIX"]
catalog, schema_name = schema_full.split(".", 1)

print(f"Schema: {catalog}.{schema_name}, Prefix: {prefix}")

# Step 1: Create a database instance (Lakebase)
db_instance_name = f"{prefix}-lakebase"
print(f"\nCreating database instance: {db_instance_name}")

try:
    inst = w.database.get_database_instance(db_instance_name)
    print(f"Instance already exists: {inst.name} ({inst.state})")
except NotFound:
    w.database.create_database_instance(
        dbsvc.DatabaseInstance(
            name=db_instance_name,
            capacity="CU_1",
        )
    )
    print("Database instance creation started.")

# Poll until available
max_wait = 600
elapsed = 0
while elapsed < max_wait:
    inst = w.database.get_database_instance(db_instance_name)
    state_str = str(inst.state)
    print(f"  Instance state ({elapsed}s): {state_str}")
    if "AVAILABLE" in state_str:
        print("Database instance is available!")
        break
    if "FAIL" in state_str:
        raise RuntimeError(f"Database instance failed: {inst}")
    time.sleep(20)
    elapsed += 20

# Step 2: Create a synced database table
synced_table_name = f"{catalog}.{schema_name}.featuresb1ea93_synced"
print(f"\nCreating synced table: {synced_table_name}")

try:
    existing_st = w.database.get_synced_database_table(synced_table_name)
    print(f"Synced table already exists: {existing_st.name}")
    print(f"Status: {existing_st.data_synchronization_status}")
except NotFound:
    spec = dbsvc.SyncedTableSpec(
        source_table_full_name=f"{catalog}.{schema_name}.featuresb1ea93",
        primary_key_columns=["row_id"],
        timeseries_key="event_time",
        scheduling_policy=dbsvc.SyncedTableSchedulingPolicy.TRIGGERED,
        create_database_objects_if_missing=True,
    )
    st = w.database.create_synced_database_table(
        dbsvc.SyncedDatabaseTable(
            name=synced_table_name,
            database_instance_name=db_instance_name,
            logical_database_name=schema_name,
            spec=spec,
        )
    )
    print(f"Synced table created: {st.name}")
    print(f"DB instance: {st.effective_database_instance_name}")
    print(f"Logical DB: {st.effective_logical_database_name}")
    print(f"UC state: {st.unity_catalog_provisioning_state}")
    print(f"Status: {st.data_synchronization_status}")

# Wait for synced table to provision
max_wait = 300
elapsed = 0
while elapsed < max_wait:
    st = w.database.get_synced_database_table(synced_table_name)
    uc_state = str(st.unity_catalog_provisioning_state)
    sync_status = st.data_synchronization_status
    detail_state = str(sync_status.detailed_state) if sync_status else None
    print(f"  UC state ({elapsed}s): {uc_state}, sync: {detail_state}")
    if "ACTIVE" in uc_state or "PROVISIONED" in uc_state:
        print("Synced table is provisioned!")
        break
    if "FAIL" in uc_state:
        raise RuntimeError(f"Synced table failed: {st}")
    time.sleep(20)
    elapsed += 20

print(f"\nDone!")
print(f"Feature table (offline): {catalog}.{schema_name}.featuresb1ea93")
print(f"Synced table (online):   {synced_table_name}")
print(f"Database instance:       {db_instance_name}")
