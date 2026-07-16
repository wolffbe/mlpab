import os, time
from datetime import timedelta
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.database import (
    DatabaseInstance, SyncedDatabaseTable, SyncedTableSpec,
    SyncedTableSchedulingPolicy, NewPipelineSpec)

w = WorkspaceClient()
catalog, schema = os.environ["MLPAB_DATABRICKS_SCHEMA"].split(".")
prefix = os.environ["MLPAB_DATABRICKS_PREFIX"]
TABLE = f"{catalog}.{schema}.incrementalf3c1bf"
ONLINE = f"{TABLE}_online"
INSTANCE = f"{prefix}-online-store"

# 1. Database instance (Lakebase) for low-latency lookup
try:
    inst = w.database.get_database_instance(INSTANCE)
    print("instance exists:", inst.name, inst.state)
except Exception:
    inst = w.database.create_database_instance_and_wait(
        DatabaseInstance(name=INSTANCE, capacity="CU_1"),
        timeout=timedelta(minutes=25))
print("instance:", inst.name, inst.state)
if str(inst.state.value) != "AVAILABLE":
    inst = w.database.wait_get_database_instance_database_available(
        INSTANCE, timeout=timedelta(minutes=25))
    print("instance now:", inst.state)

# 2. Synced table (online copy of the feature table)
def make_spec(policy, ts_key):
    return SyncedTableSpec(
        source_table_full_name=TABLE,
        primary_key_columns=["row_id"],
        timeseries_key=ts_key,
        scheduling_policy=policy,
        create_database_objects_if_missing=True,
        new_pipeline_spec=NewPipelineSpec(storage_catalog=catalog, storage_schema=schema),
    )

try:
    st = w.database.get_synced_database_table(ONLINE)
    print("synced table exists")
except Exception:
    try:
        st = w.database.create_synced_database_table(SyncedDatabaseTable(
            name=ONLINE, database_instance_name=INSTANCE,
            logical_database_name="databricks_postgres",
            spec=make_spec(SyncedTableSchedulingPolicy.SNAPSHOT, "event_time")))
        print("created synced table (SNAPSHOT + timeseries key)")
    except Exception as e:
        print("snapshot+ts failed:", e)
        st = w.database.create_synced_database_table(SyncedDatabaseTable(
            name=ONLINE, database_instance_name=INSTANCE,
            logical_database_name="databricks_postgres",
            spec=make_spec(SyncedTableSchedulingPolicy.TRIGGERED, None)))
        print("created synced table (TRIGGERED, no ts key)")

# 3. Poll until the sync is healthy
deadline = time.time() + 25 * 60
while time.time() < deadline:
    st = w.database.get_synced_database_table(ONLINE)
    s = st.data_synchronization_status
    state = s.detailed_state.value if s and s.detailed_state else None
    print("sync state:", state, flush=True)
    if state in ("ONLINE", "ONLINE_TRIGGERED_UPDATE", "ONLINE_NO_PENDING_UPDATE",
                 "ONLINE_CONTINUOUS_UPDATE", "ONLINE_UPDATING_PIPELINE_RESOURCES"):
        break
    if state in ("OFFLINE_FAILED", "ONLINE_PIPELINE_FAILED"):
        raise RuntimeError(f"sync failed: {s}")
    time.sleep(20)

print("pipeline_id:", st.data_synchronization_status.pipeline_id
      if st.data_synchronization_status else None)
print("done:", st.name)
