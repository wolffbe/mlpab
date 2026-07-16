import time
from databricks.sdk import WorkspaceClient
import databricks.sdk.service.database as db

w = WorkspaceClient()
CAT = 'workspace'
SCH = 'mlpabc1d5e2'
INST = 'mlpabc1d5e2-scaled'
SRC = f'{CAT}.{SCH}.scaledd437a3'
SYNCED = f'{CAT}.{SCH}.scaledd437a3_online'

spec = db.SyncedTableSpec(
    source_table_full_name=SRC,
    primary_key_columns=['row_id'],
    scheduling_policy=db.SyncedTableSchedulingPolicy.SNAPSHOT,
    create_database_objects_if_missing=True,
)
st = db.SyncedDatabaseTable(
    name=SYNCED,
    database_instance_name=INST,
    logical_database_name='databricks_postgres',
    spec=spec,
)
res = w.database.create_synced_database_table(st)
print('created synced table:', res.name)

# poll status until provisioned/synced
for _ in range(120):
    cur = w.database.get_synced_database_table(SYNCED)
    s = cur.data_synchronization_status
    detail = None
    if s:
        detail = s.detailed_state
    print('state:', detail)
    if detail is not None and 'ONLINE' in str(detail):
        break
    if detail is not None and 'FAILED' in str(detail):
        raise RuntimeError(f'sync failed: {s}')
    time.sleep(10)
print('final:', detail)
