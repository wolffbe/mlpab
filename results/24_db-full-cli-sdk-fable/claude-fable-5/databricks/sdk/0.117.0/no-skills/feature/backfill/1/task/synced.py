import datetime
import os
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors.platform import NotFound
from databricks.sdk.service.database import (
    DatabaseInstance,
    NewPipelineSpec,
    SyncedDatabaseTable,
    SyncedTableSchedulingPolicy,
    SyncedTableSpec,
)

SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]
CATALOG, SCHEMA_NAME = SCHEMA.split(".")
TABLE = f"{SCHEMA}.accounts06d84b"
ONLINE = f"{SCHEMA}.accounts06d84b_online"
INSTANCE = f"{PREFIX}-accounts"

w = WorkspaceClient()

existing = list(w.database.list_database_instances())
print("existing instances:", [(i.name, i.state) for i in existing])

names = [i.name for i in existing]
if INSTANCE not in names:
    print("creating database instance", INSTANCE)
    op = w.database.create_database_instance(DatabaseInstance(name=INSTANCE, capacity="CU_1"))
    inst = op.result(timeout=datetime.timedelta(minutes=25))
else:
    inst = w.database.wait_get_database_instance_database_available(
        INSTANCE, timeout=datetime.timedelta(minutes=25)
    )
print("instance state:", inst.name, inst.state)

try:
    st = w.database.get_synced_database_table(ONLINE)
    print("synced table already exists:", st.name)
except NotFound:
    st = w.database.create_synced_database_table(
        SyncedDatabaseTable(
            name=ONLINE,
            database_instance_name=INSTANCE,
            logical_database_name="databricks_postgres",
            spec=SyncedTableSpec(
                source_table_full_name=TABLE,
                primary_key_columns=["row_id"],
                timeseries_key="updated_at",
                scheduling_policy=SyncedTableSchedulingPolicy.SNAPSHOT,
                create_database_objects_if_missing=True,
                new_pipeline_spec=NewPipelineSpec(
                    storage_catalog=CATALOG, storage_schema=SCHEMA_NAME
                ),
            ),
        )
    )
    print("synced table created:", st.name)

deadline = time.time() + 25 * 60
while time.time() < deadline:
    st = w.database.get_synced_database_table(ONLINE)
    status = st.data_synchronization_status
    state = status.detailed_state if status else None
    msg = status.message if status else None
    print("state:", state, "|", msg)
    if state and "ONLINE" in str(state):
        break
    if state and "FAILED" in str(state):
        raise SystemExit(f"sync failed: {msg}")
    time.sleep(20)

print("final:", st.name, st.data_synchronization_status.detailed_state)
