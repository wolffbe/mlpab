"""Create staging table and recreate accountse81ff1 as a SyncedDatabaseTable."""
import os, time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState, ExecuteStatementRequestOnWaitTimeout
import databricks.sdk.service.database as db

w = WorkspaceClient()
schema_full = os.environ['MLPAB_DATABRICKS_SCHEMA']
catalog_name, schema_name = schema_full.split('.')
prefix = os.environ['MLPAB_DATABRICKS_PREFIX']
warehouse_id = '4dfab06c923fe3cc'
table_full = f'{catalog_name}.{schema_name}.accountse81ff1'
staging_table = f'{catalog_name}.{schema_name}.accountse81ff1_staging'
instance_name = f'{prefix}-accounts'

def run_sql(sql, timeout=180):
    resp = w.statement_execution.execute_statement(
        statement=sql,
        warehouse_id=warehouse_id,
        wait_timeout='50s',
        on_wait_timeout=ExecuteStatementRequestOnWaitTimeout.CONTINUE
    )
    stmt_id = resp.statement_id
    end = time.time() + timeout
    while time.time() < end:
        r = w.statement_execution.get_statement(stmt_id)
        if r.status.state in (
            StatementState.SUCCEEDED, StatementState.FAILED,
            StatementState.CANCELED, StatementState.CLOSED
        ):
            return r
        time.sleep(3)
    return None

# Step 1: Create staging table from existing accountse81ff1
print('Step 1: Creating staging table...')
r = run_sql(f'CREATE TABLE {staging_table} AS SELECT * FROM {table_full}')
print('  Result:', r.status.state, r.status.error or '')

# Also re-add PRIMARY KEY constraint to staging table
r2 = run_sql(f'ALTER TABLE {staging_table} ADD CONSTRAINT accountse81ff1_staging_pk PRIMARY KEY (row_id)')
print('  Add PK to staging:', r2.status.state, r2.status.error or '')

# Verify staging table has data
r3 = run_sql(f'SELECT COUNT(*) FROM {staging_table}')
print('  Staging row count:', r3.result.data_array)

# Step 2: Delete the existing _online synced table if exists (we'll try another approach)
# The staging table is now the source for the synced table

print('\nStep 2: Dropping Delta table accountse81ff1...')
r4 = run_sql(f'DROP TABLE {table_full}')
print('  Result:', r4.status.state, r4.status.error or '')

# Step 3: Create SyncedDatabaseTable with the original name
print('\nStep 3: Creating SyncedDatabaseTable accountse81ff1...')
synced_table = db.SyncedDatabaseTable(
    name=table_full,
    database_instance_name=instance_name,
    logical_database_name='feature_db',
    spec=db.SyncedTableSpec(
        source_table_full_name=staging_table,
        primary_key_columns=['row_id'],
        timeseries_key='updated_at',
        scheduling_policy=db.SyncedTableSchedulingPolicy.TRIGGERED,
        create_database_objects_if_missing=True
    )
)

try:
    result = w.database.create_synced_database_table(synced_table)
    print('Created synced table:', result.name)
    print('DB instance:', result.effective_database_instance_name)
    print('Sync status:', result.data_synchronization_status.detailed_state if result.data_synchronization_status else 'N/A')
    print('Provisioning:', result.unity_catalog_provisioning_state)
except Exception as e:
    print('Error creating synced table:', type(e).__name__, str(e)[:500])
