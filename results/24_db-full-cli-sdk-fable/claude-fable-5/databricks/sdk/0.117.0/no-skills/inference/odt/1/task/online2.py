import datetime
import inspect
import os
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import database

SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]
INSTANCE = f"{PREFIX}-online-store"

w = WorkspaceClient()

print("NewPipelineSpec:", inspect.signature(database.NewPipelineSpec.__init__))
print("SchedulingPolicy values:", list(database.SyncedTableSchedulingPolicy))

names = [i.name for i in w.database.list_database_instances()]
if INSTANCE not in names:
    print("creating instance", INSTANCE)
    inst = w.database.create_database_instance_and_wait(
        database.DatabaseInstance(name=INSTANCE, capacity="CU_1"),
        timeout=datetime.timedelta(minutes=25),
    )
else:
    print("instance exists, waiting for available")
    inst = w.database.wait_get_database_instance_database_available(
        INSTANCE, timeout=datetime.timedelta(minutes=25)
    )
print("instance state:", inst.state)

synced = database.SyncedDatabaseTable(
    name=f"{SCHEMA}.scored288ecf_online",
    database_instance_name=INSTANCE,
    logical_database_name="databricks_postgres",
    spec=database.SyncedTableSpec(
        source_table_full_name=f"{SCHEMA}.scored288ecf",
        primary_key_columns=["request_id"],
        scheduling_policy=database.SyncedTableSchedulingPolicy.SNAPSHOT,
        create_database_objects_if_missing=True,
    ),
)
res = w.database.create_synced_database_table(synced)
print("synced table created:", res.name)

deadline = time.time() + 25 * 60
while time.time() < deadline:
    st = w.database.get_synced_database_table(f"{SCHEMA}.scored288ecf_online")
    state = st.data_synchronization_status.detailed_state if st.data_synchronization_status else None
    msg = st.data_synchronization_status.message if st.data_synchronization_status else None
    print("state:", state, "|", msg)
    if state and "ONLINE" in str(state) and "PIPELINE" not in str(state):
        print("DONE - synced table online")
        break
    if state and ("FAILED" in str(state) or "ERROR" in str(state)):
        raise RuntimeError(f"synced table failed: {msg}")
    time.sleep(20)
else:
    raise RuntimeError("timed out waiting for synced table")
