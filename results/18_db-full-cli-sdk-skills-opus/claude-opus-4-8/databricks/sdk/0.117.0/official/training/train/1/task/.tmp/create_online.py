import os, time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    OnlineTable, OnlineTableSpec, OnlineTableSpecTriggeredSchedulingPolicy)
w = WorkspaceClient()
cat, sch = os.environ['MLPAB_DATABRICKS_SCHEMA'].split('.')
src = f'{cat}.{sch}.predictionsa834e5'
online_name = f'{cat}.{sch}.predictionsa834e5_online'

spec = OnlineTableSpec(
    source_table_full_name=src,
    primary_key_columns=['row_id'],
    perform_full_copy=True,
    run_triggered=OnlineTableSpecTriggeredSchedulingPolicy(),
)
try:
    ot = w.online_tables.create(table=OnlineTable(name=online_name, spec=spec))
    print('create returned:', online_name)
except Exception as e:
    print('create:', repr(e)[:300])

# poll for readiness
for i in range(40):
    try:
        cur = w.online_tables.get(name=online_name)
    except Exception as e:
        print('get err', repr(e)[:150]); time.sleep(10); continue
    state = cur.status.detailed_state if cur.status else None
    print(i, 'state:', state)
    s = str(state)
    if 'ONLINE' in s or 'ACTIVE' in s:
        print('ONLINE READY')
        break
    if 'FAILED' in s or 'OFFLINE_FAILED' in s:
        print('FAILED', cur.status)
        break
    time.sleep(15)
