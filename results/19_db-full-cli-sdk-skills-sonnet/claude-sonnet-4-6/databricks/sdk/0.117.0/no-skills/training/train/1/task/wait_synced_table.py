import os
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.postgres import SyncedTableState

w = WorkspaceClient()
prefix = os.environ['MLPAB_DATABRICKS_PREFIX']

synced_table_name = f'synced_tables/{prefix}_online_cat.pred_online_db.predictions7b586d'
pipeline_id = 'aa90a7f4-bef7-450d-9a0c-06abbc459231'

print(f'Waiting for synced table: {synced_table_name}')
print(f'Pipeline: {pipeline_id}')

# Check pipeline state
def check_pipeline():
    try:
        pl = w.pipelines.get(pipeline_id=pipeline_id)
        print(f'  Pipeline state: {pl.state}')
        return pl.state
    except Exception as e:
        print(f'  Pipeline error: {e}')
        return None

# Check synced table status
def check_synced_table():
    try:
        st = w.postgres.get_synced_table(name=synced_table_name)
        status = st.status
        state = status.detailed_state if status else None
        print(f'  Synced table state: {state}')
        if status and status.message:
            print(f'  Message: {status.message[:200]}')
        return state
    except Exception as e:
        print(f'  Synced table error: {e}')
        return None

done_states = {
    SyncedTableState.SYNCED_TABLE_ONLINE,
    SyncedTableState.SYNCED_TABLE_ONLINE_NO_PENDING_UPDATE,
    SyncedTableState.SYNCED_TABLE_ONLINE_CONTINUOUS_UPDATE,
}

error_states = {
    SyncedTableState.SYNCED_TABLE_OFFLINE,
    SyncedTableState.SYNCED_TABLE_OFFLINE_FAILED,
    SyncedTableState.SYNCED_TABLE_ONLINE_PIPELINE_FAILED,
}

max_wait = 1200
start = time.time()

while time.time() - start < max_wait:
    print(f'\n[{int(time.time() - start)}s elapsed]')
    state = check_synced_table()
    check_pipeline()

    if state in done_states:
        print('Synced table is ONLINE!')
        break
    elif state in error_states:
        print('Synced table error!')
        break


    print('Waiting 30s...')
    time.sleep(30)
else:
    print('Timeout waiting for synced table')

print('Done checking.')
