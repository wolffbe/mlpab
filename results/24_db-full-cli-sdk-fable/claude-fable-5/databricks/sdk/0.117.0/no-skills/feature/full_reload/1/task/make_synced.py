import datetime
import os
import time

from databricks.sdk import WorkspaceClient
import databricks.sdk.service.database as db

FULL_SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]
CATALOG, SCHEMA = FULL_SCHEMA.split(".")
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]
SOURCE = f"{FULL_SCHEMA}.customers03eedc"
ONLINE = f"{FULL_SCHEMA}.customers03eedc_online"
INSTANCE = f"{PREFIX}-lakebase"

w = WorkspaceClient()

existing = [i for i in w.database.list_database_instances() if i.name == INSTANCE]
if existing:
    inst = existing[0]
    print("instance exists:", inst.name, inst.state)
else:
    inst = w.database.create_database_instance_and_wait(
        db.DatabaseInstance(name=INSTANCE, capacity="CU_1"),
        timeout=datetime.timedelta(minutes=25),
    )
    print("instance created:", inst.name, inst.state)

synced = w.database.create_synced_database_table(
    db.SyncedDatabaseTable(
        name=ONLINE,
        database_instance_name=INSTANCE,
        logical_database_name="customers03eedc_db",
        spec=db.SyncedTableSpec(
            source_table_full_name=SOURCE,
            primary_key_columns=["row_id"],
            timeseries_key="updated_at",
            scheduling_policy=db.SyncedTableSchedulingPolicy.SNAPSHOT,
            create_database_objects_if_missing=True,
            new_pipeline_spec=db.NewPipelineSpec(
                storage_catalog=CATALOG, storage_schema=SCHEMA
            ),
        ),
    )
)
print("synced table create requested:", synced.name)

deadline = time.time() + 25 * 60
while time.time() < deadline:
    st = w.database.get_synced_database_table(ONLINE)
    status = st.data_synchronization_status
    detailed = status.detailed_state if status else None
    print("state:", detailed, "| provisioning:", st.unity_catalog_provisioning_state)
    if detailed and "ONLINE" in str(detailed):
        print("SYNCED TABLE READY")
        break
    if detailed and ("FAILED" in str(detailed) or "ERROR" in str(detailed)):
        print("SYNC FAILED:", status)
        break
    time.sleep(20)
