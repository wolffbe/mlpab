import datetime
import os
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.database import (
    DatabaseInstance,
    SyncedDatabaseTable,
    SyncedTableSpec,
    SyncedTableSchedulingPolicy,
)

w = WorkspaceClient()
cat, sch = os.environ['MLPAB_DATABRICKS_SCHEMA'].split('.')
prefix = os.environ['MLPAB_DATABRICKS_PREFIX']
FQN = f'{cat}.{sch}.scaledf9e607'
ONLINE = f'{cat}.{sch}.scaledf9e607_online'
INSTANCE = f'{prefix}-online-store'

try:
    inst = w.database.get_database_instance(name=INSTANCE)
    print('instance exists:', inst.state)
except Exception:
    waiter = w.database.create_database_instance(
        DatabaseInstance(name=INSTANCE, capacity='CU_1')
    )
    inst = waiter.result(timeout=datetime.timedelta(minutes=25))
    print('instance created:', inst.state)

st = w.database.create_synced_database_table(
    SyncedDatabaseTable(
        name=ONLINE,
        database_instance_name=INSTANCE,
        logical_database_name='online',
        spec=SyncedTableSpec(
            source_table_full_name=FQN,
            primary_key_columns=['row_id'],
            scheduling_policy=SyncedTableSchedulingPolicy.SNAPSHOT,
            create_database_objects_if_missing=True,
        ),
    )
)
print('synced table created:', st.name)

deadline = time.time() + 25 * 60
while time.time() < deadline:
    cur = w.database.get_synced_database_table(name=ONLINE)
    s = cur.data_synchronization_status
    state = s.detailed_state.value if s and s.detailed_state else 'UNKNOWN'
    print('state:', state, flush=True)
    if state in ('SYNCED_TABLE_ONLINE', 'ONLINE'):
        break
    if 'FAILED' in state or 'ERROR' in state:
        print('failure detail:', s)
        break
    time.sleep(20)
print('done')
