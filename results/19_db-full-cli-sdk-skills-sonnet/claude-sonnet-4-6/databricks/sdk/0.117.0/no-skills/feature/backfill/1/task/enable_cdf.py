"""Enable Change Data Feed on staging table and retry synced table creation."""
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

# Enable CDF on staging table
print('Enabling CDF on staging table...')
cdf_sql = (
    "ALTER TABLE " + staging_table + " "
    + "SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')"
)
r = run_sql(cdf_sql)
print('  CDF result:', r.status.state, r.status.error or '')

# Create SyncedDatabaseTable with the original name
print('\nCreating SyncedDatabaseTable accountse81ff1...')
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
    print('DB instance:', result.effective_database_instance_name or 'provisioning...')
    status = result.data_synchronization_status
    if status:
        print('Sync status:', status.detailed_state)
        print('Message:', status.message)
    print('Provisioning:', result.unity_catalog_provisioning_state)
except Exception as e:
    print('Error:', type(e).__name__, str(e)[:600])
