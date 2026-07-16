import os, time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.database import (
    SyncedDatabaseTable, SyncedTableSpec, SyncedTableSchedulingPolicy, NewPipelineSpec)
w = WorkspaceClient()
cat, sch = os.environ['MLPAB_DATABRICKS_SCHEMA'].split('.')
prefix = os.environ['MLPAB_DATABRICKS_PREFIX']
inst_name = f'{prefix}-lba834e5'
src = f'{cat}.{sch}.predictionsa834e5'
synced_name = f'{cat}.{sch}.predictionsa834e5_synced'

spec = SyncedTableSpec(
    source_table_full_name=src,
    primary_key_columns=['row_id'],
    scheduling_policy=SyncedTableSchedulingPolicy.SNAPSHOT,
    create_database_objects_if_missing=True,
    new_pipeline_spec=NewPipelineSpec(storage_catalog=cat, storage_schema=sch),
)
sdt = SyncedDatabaseTable(
    name=synced_name,
    database_instance_name=inst_name,
    logical_database_name='databricks_postgres',
    spec=spec,
)
try:
    res = w.database.create_synced_database_table(sdt)
    print('created synced table:', res.name)
except Exception as e:
    print('create:', repr(e)[:400])

for i in range(60):
    cur = w.database.get_synced_database_table(name=synced_name)
    prov = cur.unity_catalog_provisioning_state
    dss = cur.data_synchronization_status
    detail = None
    if dss:
        detail = getattr(dss, 'detailed_state', None)
    print(i, 'uc_prov:', prov, 'sync_detail:', detail)
    s = str(prov) + str(detail)
    if 'ONLINE' in str(detail) or ('ACTIVE' in str(prov) and detail and 'ONLINE' in str(detail)):
        print('SYNCED ONLINE READY'); break
    if 'FAIL' in s.upper():
        print('FAILED', dss); break
    time.sleep(15)
