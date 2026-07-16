import time
import databricks.sdk

sdk = databricks.sdk
db = sdk.service.database
w = sdk.WorkspaceClient()
SCHEMA = "workspace.mlpab4fd108"
INSTANCE = "mlpab4fd108-online-store"

inst = w.database.create_database_instance_and_wait(
    db.DatabaseInstance(name=INSTANCE, capacity="CU_1")
)
print("instance:", inst.name, inst.state)

st = db.SyncedDatabaseTable(
    name=f"{SCHEMA}.scoresbedd56_online",
    database_instance_name=INSTANCE,
    logical_database_name="scores_db",
    spec=db.SyncedTableSpec(
        source_table_full_name=f"{SCHEMA}.scoresbedd56",
        primary_key_columns=["account_id"],
        scheduling_policy=db.SyncedTableSchedulingPolicy.SNAPSHOT,
        create_database_objects_if_missing=True,
        new_pipeline_spec=db.NewPipelineSpec(
            storage_catalog="workspace", storage_schema="mlpab4fd108"
        ),
    ),
)
created = w.database.create_synced_database_table(st)
print("synced table created:", created.name)

for i in range(120):
    cur = w.database.get_synced_database_table(f"{SCHEMA}.scoresbedd56_online")
    state = cur.data_synchronization_status.detailed_state if cur.data_synchronization_status else None
    msg = cur.data_synchronization_status.message if cur.data_synchronization_status else None
    print(f"[{i}] state={state} msg={msg}")
    if state and str(state) in (
        "SyncedTableState.SYNCED_TABLE_ONLINE",
        "SyncedTableState.ONLINE",
        "SyncedTableState.SYNCED_TABLE_ONLINE_NO_PENDING_UPDATE",
        "SyncedTableState.ONLINE_NO_PENDING_UPDATE",
    ):
        print("ONLINE")
        break
    if state and "FAILED" in str(state):
        raise RuntimeError(f"sync failed: {msg}")
    time.sleep(15)
