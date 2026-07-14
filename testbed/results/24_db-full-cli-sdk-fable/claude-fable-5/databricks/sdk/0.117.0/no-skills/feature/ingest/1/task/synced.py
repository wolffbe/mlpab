import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.database import (
    DatabaseInstance,
    SyncedDatabaseTable,
    SyncedTableSpec,
    NewPipelineSpec,
    SyncedTableSchedulingPolicy,
)

w = WorkspaceClient()
CAT, SCH = "workspace", "mlpabce02a9"
FQ = f"{CAT}.{SCH}"
INSTANCE = "mlpabce02a9-online-store"

# 1. Database instance (prefixed with run prefix)
try:
    inst = w.database.get_database_instance(INSTANCE)
    print("instance exists:", inst.state)
except Exception:
    print("creating instance...")
    inst = w.database.create_database_instance_and_wait(
        DatabaseInstance(name=INSTANCE, capacity="CU_1")
    )
    print("instance:", inst.state)

# wait until AVAILABLE
deadline = time.time() + 1800
while time.time() < deadline:
    inst = w.database.get_database_instance(INSTANCE)
    if str(inst.state).endswith("AVAILABLE"):
        break
    print("instance state:", inst.state)
    time.sleep(20)
print("instance ready:", inst.state)

# 2. Synced table for low-latency lookup
st = SyncedDatabaseTable(
    name=f"{FQ}.transactionsf10ad0_online",
    database_instance_name=INSTANCE,
    logical_database_name="transactions",
    spec=SyncedTableSpec(
        source_table_full_name=f"{FQ}.transactionsf10ad0",
        primary_key_columns=["row_id"],
        timeseries_key="event_time",
        scheduling_policy=SyncedTableSchedulingPolicy.SNAPSHOT,
        create_database_objects_if_missing=True,
        new_pipeline_spec=NewPipelineSpec(storage_catalog=CAT, storage_schema=SCH),
    ),
)
try:
    res = w.database.create_synced_database_table(st)
    print("synced table created:", res.name)
except Exception as e:
    if "timeseries" in str(e).lower():
        print("retrying without timeseries_key:", e)
        st.spec.timeseries_key = None
        res = w.database.create_synced_database_table(st)
        print("synced table created:", res.name)
    else:
        raise

# 3. Wait for sync to complete
deadline = time.time() + 1800
while time.time() < deadline:
    cur = w.database.get_synced_database_table(f"{FQ}.transactionsf10ad0_online")
    s = cur.data_synchronization_status
    state = s.detailed_state if s else None
    print("sync state:", state)
    if state and any(k in str(state) for k in ("ONLINE", "FAILED")) and "PIPELINE" not in str(state) and "UPDATING" not in str(state):
        break
    time.sleep(20)
print("final:", state)
if s and s.message:
    print("message:", s.message)
