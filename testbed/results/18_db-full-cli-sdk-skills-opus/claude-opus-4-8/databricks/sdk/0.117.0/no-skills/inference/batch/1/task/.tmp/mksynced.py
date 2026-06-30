import time
from databricks.sdk import WorkspaceClient
import databricks.sdk.service.database as db

w = WorkspaceClient()

synced_name = 'workspace.mlpabc69812.scores3380ed_online'
spec = db.SyncedTableSpec(
    source_table_full_name='workspace.mlpabc69812.scores3380ed',
    primary_key_columns=['account_id'],
    scheduling_policy=db.SyncedTableSchedulingPolicy.SNAPSHOT,
    create_database_objects_if_missing=True,
)
st = db.SyncedDatabaseTable(
    name=synced_name,
    database_instance_name='mlpabc69812-scores3380ed-lb',
    logical_database_name='databricks_postgres',
    spec=spec,
)
res = w.database.create_synced_database_table(st)
print('created synced table:', res.name)

# poll status until ready
for _ in range(60):
    cur = w.database.get_synced_database_table(synced_name)
    status = cur.data_synchronization_status
    detail = None
    if status:
        detail = status.detailed_state
    print('state:', detail)
    if detail and 'ONLINE' in str(detail):
        break
    if detail and 'FAILED' in str(detail):
        print('FAILED:', status)
        break
    time.sleep(15)
