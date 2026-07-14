import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import ExecuteStatementRequestOnWaitTimeout

w = WorkspaceClient()
PREFIX = 'mlpab483bdc'
DB_CATALOG_NAME = f'{PREFIX}_dbcat'
TABLE_NAME = 'profilesaa70e4'
SYNCED_TABLE_NAME = f'{DB_CATALOG_NAME}.public.{TABLE_NAME}'
WAREHOUSE_ID = '4dfab06c923fe3cc'

def exec_sql(sql):
    resp = w.statement_execution.execute_statement(
        statement=sql,
        warehouse_id=WAREHOUSE_ID,
        wait_timeout='50s',
        on_wait_timeout=ExecuteStatementRequestOnWaitTimeout.CONTINUE,
    )
    stmt_id = resp.statement_id
    while resp.status.state.value in ('PENDING', 'RUNNING'):
        time.sleep(2)
        resp = w.statement_execution.get_statement(stmt_id)
    if resp.status.state.value != 'SUCCEEDED':
        raise RuntimeError(f'SQL failed: {resp.status.error}')
    return resp

# Count rows
resp = exec_sql(f'SELECT COUNT(*) FROM {SYNCED_TABLE_NAME}')
print(f'Row count: {resp.result}')

# Sample
resp2 = exec_sql(f'SELECT * FROM {SYNCED_TABLE_NAME} LIMIT 3')
print(f'Sample rows: {resp2.result.data_array}')
