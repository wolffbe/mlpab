"""Wait for synced table to become online."""
import os, time
from databricks.sdk import WorkspaceClient
import databricks.sdk.service.database as db

w = WorkspaceClient()
schema_full = os.environ['MLPAB_DATABRICKS_SCHEMA']
catalog_name, schema_name = schema_full.split('.')
table_full = f'{catalog_name}.{schema_name}.accountse81ff1'

ONLINE_STATES = {
    db.SyncedTableState.SYNCED_TABLE_ONLINE,
    db.SyncedTableState.SYNCED_TABLE_ONLINE_CONTINUOUS_UPDATE,
    db.SyncedTableState.SYNCED_TABLE_ONLINE_NO_PENDING_UPDATE,
    db.SyncedTableState.SYNCED_TABLE_ONLINE_TRIGGERED_UPDATE,
    db.SyncedTableState.SYNCED_TABLE_ONLINE_UPDATING_PIPELINE_RESOURCES,
}
FAIL_STATES = {
    db.SyncedTableState.SYNCED_TABLED_OFFLINE,
    db.SyncedTableState.SYNCED_TABLE_OFFLINE_FAILED,
    db.SyncedTableState.SYNCED_TABLE_ONLINE_PIPELINE_FAILED,
}

print('Waiting for synced table to come online...')
for i in range(60):
    result = w.database.get_synced_database_table(table_full)
    status = result.data_synchronization_status
    state = status.detailed_state if status else None
    print(f'  [{i*15}s] State: {state}')

    if state in ONLINE_STATES:
        print('Synced table is online!')
        break
    elif state in FAIL_STATES:
        print('Synced table failed:', status)
        break
    time.sleep(15)

# Final check
result = w.database.get_synced_database_table(table_full)
print('\nFinal state:', result.data_synchronization_status.detailed_state if result.data_synchronization_status else 'N/A')
print('UC provisioning:', result.unity_catalog_provisioning_state)
print('DB instance:', result.effective_database_instance_name)
print('Logical DB:', result.effective_logical_database_name)
if result.spec:
    print('Source table:', result.spec.source_table_full_name)
    print('PK columns:', result.spec.primary_key_columns)
    print('TS key:', result.spec.timeseries_key)
