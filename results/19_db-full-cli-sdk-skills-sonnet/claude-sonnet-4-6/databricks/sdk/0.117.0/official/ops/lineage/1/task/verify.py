import os, time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

w = WorkspaceClient()
schema = os.environ['MLPAB_DATABRICKS_SCHEMA']
catalog, schema_name = schema.split('.')
WAREHOUSE_ID = '4dfab06c923fe3cc'

def run_sql(statement):
    resp = w.statement_execution.execute_statement(
        statement=statement, warehouse_id=WAREHOUSE_ID, wait_timeout='50s')
    terminal = {StatementState.SUCCEEDED, StatementState.FAILED, StatementState.CANCELED, StatementState.CLOSED}
    while resp.status.state not in terminal:
        time.sleep(2)
        resp = w.statement_execution.get_statement(resp.statement_id)
    if resp.status.state not in [StatementState.SUCCEEDED]:
        raise Exception(f'SQL failed: {resp.status.error}')
    return resp

# Verify row counts
for table in ['rawad05474', 'rawbd05474', 'derivedd05474']:
    resp = run_sql(f'SELECT COUNT(*) as cnt FROM {catalog}.{schema_name}.{table}')
    rows = resp.result.data_array
    print(f'{table}: {rows[0][0]} rows')

# Sample derived table
resp = run_sql(f'SELECT * FROM {catalog}.{schema_name}.derivedd05474 ORDER BY row_id LIMIT 5')
print('derivedd05474 sample:')
for row in resp.result.data_array:
    print(f'  {row}')

# Verify lineage
lineage = w.api_client.do('GET', '/api/2.0/lineage-tracking/table-lineage',
    query={'table_name': f'{catalog}.{schema_name}.derivedd05474'})
upstream_tables = sorted([u['tableInfo']['name'] for u in lineage.get('upstreams', [])])
print(f'Lineage upstreams: {upstream_tables}')

# Check submission files
import os as _os
print('\nSubmission files:')
for f in _os.listdir('submission'):
    print(f'  {f}')

import json
with open('submission/answers.json') as f:
    print(f'answers.json: {json.load(f)}')
