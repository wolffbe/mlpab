import os
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.postgres import (
    SyncedTable,
    SyncedTableSyncedTableSpec,
    SyncedTableSyncedTableSpecSyncedTableSchedulingPolicy,
    NewPipelineSpec,
    SyncedTableState,
)

w = WorkspaceClient()
prefix = os.environ['MLPAB_DATABRICKS_PREFIX']
schema = os.environ['MLPAB_DATABRICKS_SCHEMA']
catalog_name, schema_name = schema.split('.')

source_table = f'{schema}.predictions7b586d'
dest_catalog = f'{prefix}_online_cat'
synced_table_id = f'{dest_catalog}.pred_online_db.predictions7b586d'
synced_table_name = f'synced_tables/{synced_table_id}'

print(f'Creating postgres synced table...')
print(f'Source: {source_table}')
print(f'Destination: {synced_table_id}')

try:
    op = w.postgres.create_synced_table(
        synced_table=SyncedTable(
            spec=SyncedTableSyncedTableSpec(
                source_table_full_name=source_table,
                primary_key_columns=['row_id'],
                scheduling_policy=SyncedTableSyncedTableSpecSyncedTableSchedulingPolicy.SNAPSHOT,
                create_database_objects_if_missing=True,
                postgres_database='pred_online_db',
                new_pipeline_spec=NewPipelineSpec(
                    storage_catalog=catalog_name,
                    storage_schema=schema_name,
                ),
            ),
        ),
        synced_table_id=synced_table_id,
    )
    print('Operation initiated:', op)
except Exception as e:
    print('Create error (checking if table was created anyway):', str(e)[:200])

# Wait a moment for async creation
time.sleep(5)

# Check if the table was created
print()
print('Checking table status...')
for attempt in range(40):
    try:
        st = w.postgres.get_synced_table(name=synced_table_name)
        state = st.status.detailed_state if st.status else None
        print(f'  [{attempt*30}s] State: {state}')
        if st.status and st.status.message:
            print(f'  Message: {st.status.message[:200]}')
        if state == SyncedTableState.SYNCED_TABLE_ONLINE_NO_PENDING_UPDATE:
            print('ONLINE and synced!')
            break
        elif state == SyncedTableState.SYNCED_TABLE_ONLINE:
            print('ONLINE!')
            break
        elif state in (SyncedTableState.SYNCED_TABLE_OFFLINE_FAILED, SyncedTableState.SYNCED_TABLE_ONLINE_PIPELINE_FAILED):
            print('Failed!')
            break
        time.sleep(30)
    except Exception as e:
        print(f'  [{attempt*30}s] Error: {e}')
        break

print('Done.')
