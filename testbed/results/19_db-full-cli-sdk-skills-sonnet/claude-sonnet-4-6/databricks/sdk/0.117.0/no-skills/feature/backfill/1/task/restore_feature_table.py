"""Restore accountse81ff1 as proper Delta feature table with PK and CDF, then set up online table."""
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
online_table = f'{catalog_name}.{schema_name}.accountse81ff1_online'
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

# Step 1: Delete the SyncedDatabaseTable accountse81ff1
print('Step 1: Deleting SyncedDatabaseTable accountse81ff1...')
try:
    w.database.delete_synced_database_table(table_full)
    print('  Deleted SyncedDatabaseTable')
except Exception as e:
    print('  Error:', type(e).__name__, str(e)[:200])

# Step 2: Also delete the orphaned accountse81ff1_online
print('Step 2: Deleting orphaned accountse81ff1_online...')
try:
    w.database.delete_synced_database_table(online_table)
    print('  Deleted accountse81ff1_online')
except Exception as e:
    print('  Error:', type(e).__name__, str(e)[:200])

# Step 3: Create the Delta feature table with NOT NULL row_id and PRIMARY KEY
print('Step 3: Creating Delta feature table accountse81ff1...')
create_sql = (
    "CREATE TABLE " + table_full + " ("
    "  row_id STRING NOT NULL,"
    "  status STRING,"
    "  balance DOUBLE,"
    "  updated_at BIGINT,"
    "  CONSTRAINT accountse81ff1_pk PRIMARY KEY (row_id)"
    ") TBLPROPERTIES ("
    "  'delta.enableChangeDataFeed' = 'true'"
    ")"
)
r = run_sql(create_sql)
print('  Create result:', r.status.state, r.status.error or '')

# Step 4: Insert data from staging table
print('Step 4: Inserting data from staging...')
r2 = run_sql(f'INSERT INTO {table_full} SELECT row_id, status, balance, updated_at FROM {staging_table}')
print('  Insert result:', r2.status.state, r2.status.error or '')

# Verify count
r3 = run_sql(f'SELECT COUNT(*) FROM {table_full}')
print('  Row count:', r3.result.data_array if r3.result else 'N/A')

# Step 5: Set event-time property
print('Step 5: Setting event-time property...')
ts_sql = (
    "ALTER TABLE " + table_full + " "
    + "SET TBLPROPERTIES ('databricks.feature.timeseries_key' = 'updated_at')"
)
r4 = run_sql(ts_sql)
print('  Set TS key result:', r4.status.state, r4.status.error or '')

# Step 6: Create SyncedDatabaseTable accountse81ff1_online
print('Step 6: Creating SyncedDatabaseTable accountse81ff1_online...')
synced = db.SyncedDatabaseTable(
    name=online_table,
    database_instance_name=instance_name,
    logical_database_name='feature_db',
    spec=db.SyncedTableSpec(
        source_table_full_name=table_full,
        primary_key_columns=['row_id'],
        timeseries_key='updated_at',
        scheduling_policy=db.SyncedTableSchedulingPolicy.TRIGGERED,
        create_database_objects_if_missing=True
    )
)
try:
    result = w.database.create_synced_database_table(synced)
    print('  Created:', result.name)
    print('  DB instance:', result.effective_database_instance_name)
    status = result.data_synchronization_status
    if status:
        print('  Sync status:', status.detailed_state)
        print('  Message:', status.message)
except Exception as e:
    print('  Error:', type(e).__name__, str(e)[:600])
