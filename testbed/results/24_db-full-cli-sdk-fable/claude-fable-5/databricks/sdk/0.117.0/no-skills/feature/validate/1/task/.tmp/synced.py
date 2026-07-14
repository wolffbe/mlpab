import datetime
import os
import time

from databricks.sdk import WorkspaceClient
import databricks.sdk.service.database as db

w = WorkspaceClient()
SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]
CATALOG, SCHEMA_NAME = SCHEMA.split(".")
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]
INSTANCE = f"{PREFIX}-lakebase"
SRC = f"{SCHEMA}.events385469"
SYNCED = f"{SCHEMA}.events385469_online"

# 1. Database instance
try:
    inst = w.database.get_database_instance(INSTANCE)
    print("instance exists:", inst.state)
except Exception:
    print("creating instance", INSTANCE)
    inst = w.database.create_database_instance_and_wait(
        db.DatabaseInstance(name=INSTANCE, capacity="CU_1"),
        timeout=datetime.timedelta(minutes=25),
    )
    print("instance state:", inst.state)

# 2. Synced table
spec = db.SyncedTableSpec(
    source_table_full_name=SRC,
    primary_key_columns=["row_id"],
    scheduling_policy=db.SyncedTableSchedulingPolicy.SNAPSHOT,
    create_database_objects_if_missing=True,
    new_pipeline_spec=db.NewPipelineSpec(
        storage_catalog=CATALOG, storage_schema=SCHEMA_NAME
    ),
)
st = w.database.create_synced_database_table(
    db.SyncedDatabaseTable(
        name=SYNCED,
        database_instance_name=INSTANCE,
        logical_database_name=SCHEMA_NAME,
        spec=spec,
    )
)
print("synced table created:", st.name)

# 3. Wait for sync
deadline = time.time() + 25 * 60
while time.time() < deadline:
    cur = w.database.get_synced_database_table(SYNCED)
    status = cur.data_synchronization_status
    state = status.detailed_state if status else None
    print("state:", state)
    s = str(state)
    if "ONLINE" in s and "PIPELINE" not in s and "UPDATE" not in s:
        print("SYNCED OK")
        break
    if "FAILED" in s or "ERROR" in s:
        print("FAILED:", status.message if status else None)
        break
    time.sleep(20)
else:
    print("timed out waiting for sync")
