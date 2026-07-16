import datetime, os, time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.database import (
    DatabaseInstance,
    SyncedDatabaseTable,
    SyncedTableSpec,
    SyncedTableSchedulingPolicy,
)

w = WorkspaceClient()
catalog, schema = os.environ["MLPAB_DATABRICKS_SCHEMA"].split(".")
prefix = os.environ["MLPAB_DATABRICKS_PREFIX"]
instance_name = f"{prefix}-lakebase"
src = f"{catalog}.{schema}.predictions178367"
synced_name = f"{catalog}.{schema}.predictions178367_online"

existing = {i.name: i for i in w.database.list_database_instances()}
print("existing instances:", list(existing))
if instance_name in existing:
    inst = existing[instance_name]
else:
    inst = w.database.create_database_instance_and_wait(
        DatabaseInstance(name=instance_name, capacity="CU_1"),
        timeout=datetime.timedelta(minutes=30),
    )
print("instance:", inst.name, inst.state)

st = w.database.create_synced_database_table(
    SyncedDatabaseTable(
        name=synced_name,
        database_instance_name=instance_name,
        logical_database_name="predictions_db",
        spec=SyncedTableSpec(
            source_table_full_name=src,
            primary_key_columns=["row_id"],
            scheduling_policy=SyncedTableSchedulingPolicy.SNAPSHOT,
            create_database_objects_if_missing=True,
        ),
    )
)
print("synced table created:", st.name)

deadline = time.time() + 1800
while time.time() < deadline:
    cur = w.database.get_synced_database_table(synced_name)
    s = cur.data_synchronization_status
    detailed = s.detailed_state.value if s and s.detailed_state else None
    msg = s.message if s else None
    print("state:", detailed, "|", msg)
    if detailed and ("ONLINE" in detailed and "PIPELINE" not in detailed or detailed in ("ONLINE", "ONLINE_NO_PENDING_UPDATE")):
        break
    if detailed and ("FAILED" in detailed or "ERROR" in detailed):
        raise RuntimeError(f"sync failed: {detailed} {msg}")
    time.sleep(20)
print("done")
