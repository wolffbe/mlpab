from databricks.sdk import WorkspaceClient
from databricks.sdk.service import database
import os, time

w = WorkspaceClient()
PREFIX = os.environ['MLPAB_DATABRICKS_PREFIX']
inst_name = f"{PREFIX}-ccfraud"
SCHEMA = "workspace.mlpabea3b07"
src = f"{SCHEMA}.ccpred2dbe0a"
synced_name = f"{SCHEMA}.ccpred2dbe0a_online"

spec = database.SyncedTableSpec(
    source_table_full_name=src,
    primary_key_columns=["transaction_id"],
    scheduling_policy=database.SyncedTableSchedulingPolicy.SNAPSHOT,
    create_database_objects_if_missing=True,
    new_pipeline_spec=database.NewPipelineSpec(storage_catalog="workspace", storage_schema="mlpabea3b07"),
)
st = database.SyncedDatabaseTable(
    name=synced_name,
    database_instance_name=inst_name,
    logical_database_name="databricks_postgres",
    spec=spec,
)
try:
    res = w.database.create_synced_database_table(st)
    print("synced table created:", res.name, flush=True)
except Exception as e:
    print("create err:", str(e)[:600], flush=True)

# poll for sync state
for i in range(60):
    cur = w.database.get_synced_database_table(synced_name)
    dss = cur.data_synchronization_status
    state = dss.detailed_state if dss else None
    print(i, "prov", cur.unity_catalog_provisioning_state, "sync", state, flush=True)
    if state and "ONLINE" in str(state):
        break
    if str(cur.unity_catalog_provisioning_state) and "FAILED" in str(cur.unity_catalog_provisioning_state):
        break
    time.sleep(15)
print("DONE")
