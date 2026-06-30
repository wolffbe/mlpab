import os, time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState, ExecuteStatementRequestOnWaitTimeout

w = WorkspaceClient()
schema_full = os.environ['MLPAB_DATABRICKS_SCHEMA']
catalog_name, schema_name = schema_full.split('.')
warehouse_id = '4dfab06c923fe3cc'
table_full = f'{catalog_name}.{schema_name}.accountse81ff1'

def run_sql(sql, timeout=120):
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
        if r.status.state in (StatementState.SUCCEEDED, StatementState.FAILED, StatementState.CANCELED, StatementState.CLOSED):
            return r
        time.sleep(2)
    return None

# Set event time property via TBLPROPERTIES
prop_sql = (
    "ALTER TABLE " + table_full + " "
    + "SET TBLPROPERTIES ('databricks.feature.timeseries_key' = 'updated_at')"
)
r = run_sql(prop_sql)
print('State:', r.status.state)
if r.status.error:
    print('Error:', r.status.error)

# Show TBLPROPERTIES
r2 = run_sql(f'SHOW TBLPROPERTIES {table_full}')
if r2.result and r2.result.data_array:
    for row in r2.result.data_array:
        if 'feature' in str(row).lower() or 'timeseries' in str(row).lower():
            print('Feature prop:', row)
