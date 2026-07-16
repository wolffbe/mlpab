from databricks.sdk import WorkspaceClient
from databricks.sdk.service import database as db
import time
w = WorkspaceClient()
C = 'workspace.mlpab6d0586'
INST = 'mlpab6d0586-lakebase'
SRC = f"{C}.events88b330"
ONLINE = f"{C}.events88b330_online"

spec = db.SyncedTableSpec(
    source_table_full_name=SRC,
    primary_key_columns=['row_id'],
    timeseries_key='event_time',
    scheduling_policy=db.SyncedTableSchedulingPolicy.SNAPSHOT,
    create_database_objects_if_missing=True,
    new_pipeline_spec=db.NewPipelineSpec(storage_catalog='workspace', storage_schema='mlpab6d0586'),
)
st = db.SyncedDatabaseTable(name=ONLINE, database_instance_name=INST,
                            logical_database_name='databricks_postgres', spec=spec)
try:
    res = w.database.create_synced_database_table(synced_table=st)
    print('created synced table:', res.name)
except Exception as e:
    print('create error:', repr(e))
    res = w.database.get_synced_database_table(name=ONLINE)

# poll until synced
for _ in range(120):
    cur = w.database.get_synced_database_table(name=ONLINE)
    prov = cur.unity_catalog_provisioning_state
    dss = cur.data_synchronization_status
    detail = getattr(dss, 'detailed_state', None) if dss else None
    print('prov:', prov, 'sync:', detail)
    if str(prov) and 'ACTIVE' in str(prov):
        if detail is None or 'ONLINE' in str(detail) or 'SUCCEED' in str(detail) or 'TRIGGERED' in str(detail):
            print('READY')
            break
    time.sleep(15)
print('final prov:', cur.unity_catalog_provisioning_state, 'sync:', getattr(cur.data_synchronization_status,'detailed_state',None) if cur.data_synchronization_status else None)
