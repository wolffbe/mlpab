import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.database import (
    DatabaseInstance, SyncedDatabaseTable, SyncedTableSpec,
    SyncedTableSchedulingPolicy, NewPipelineSpec)

w = WorkspaceClient()
schema = "workspace.mlpab60c44c"
INSTANCE = "mlpab60c44c-store"

names = [i.name for i in w.database.list_database_instances()]
if INSTANCE not in names:
    inst = w.database.create_database_instance_and_wait(
        DatabaseInstance(name=INSTANCE, capacity="CU_1"),
        timeout=__import__("datetime").timedelta(minutes=25))
    print("instance created:", inst.name, inst.state)
else:
    print("instance exists")

st = SyncedDatabaseTable(
    name=f"{schema}.features3bde51_online",
    database_instance_name=INSTANCE,
    logical_database_name="featdb",
    spec=SyncedTableSpec(
        source_table_full_name=f"{schema}.features3bde51",
        primary_key_columns=["row_id"],
        scheduling_policy=SyncedTableSchedulingPolicy.SNAPSHOT,
        create_database_objects_if_missing=True,
        new_pipeline_spec=NewPipelineSpec(
            storage_catalog="workspace", storage_schema="mlpab60c44c"),
    ),
)
res = w.database.create_synced_database_table(st)
print("synced table created:", res.name)

deadline = time.time() + 1800
while time.time() < deadline:
    cur = w.database.get_synced_database_table(f"{schema}.features3bde51_online")
    s = cur.data_synchronization_status
    state = s.detailed_state if s else None
    print("state:", state, "|", (s.message if s else ""))
    if state and "ONLINE" in str(state) and "OFFLINE" not in str(state):
        print("DONE")
        break
    if state and "FAILED" in str(state):
        raise RuntimeError(f"sync failed: {s.message}")
    time.sleep(20)
