#!/usr/bin/env python3
"""Final verification of the feature table and synced table."""
import os
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState, ExecuteStatementRequestOnWaitTimeout

w = WorkspaceClient()
schema_full = os.environ['MLPAB_DATABRICKS_SCHEMA']
catalog_name, schema_name = schema_full.split('.')
full_table_name = f'{catalog_name}.{schema_name}.scored50223c'
synced_table_name = f'{catalog_name}.{schema_name}.scored50223c_synced'

# Get warehouse
warehouses = list(w.warehouses.list())
warehouse_id = warehouses[0].id

def run_sql(sql):
    stmt = w.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=sql,
        wait_timeout='50s',
        on_wait_timeout=ExecuteStatementRequestOnWaitTimeout.CONTINUE
    )
    start = time.time()
    while stmt.status.state in [StatementState.PENDING, StatementState.RUNNING]:
        if int(time.time() - start) > 60:
            return None, 'timeout'
        time.sleep(3)
        stmt = w.statement_execution.get_statement(statement_id=stmt.statement_id)
    if stmt.status.error:
        return None, str(stmt.status.error)
    return stmt.result, None

print('=== Feature Table Verification ===')

# Check row count
result, err = run_sql(f"SELECT COUNT(*) as cnt FROM {full_table_name}")
print(f'Total rows: {result.data_array if result else err}')

# Check distinct request_ids
result, err = run_sql(f"SELECT COUNT(DISTINCT request_id) as dist FROM {full_table_name}")
print(f'Distinct request_ids: {result.data_array if result else err}')

# Check sample data
result, err = run_sql(f"SELECT request_id, account_id, distance_deg, score FROM {full_table_name} ORDER BY request_id LIMIT 3")
print('Sample rows:')
if result and result.data_array:
    for row in result.data_array:
        print(f'  {row}')

# Check min/max of scores
result, err = run_sql(f"SELECT MIN(distance_deg), MAX(distance_deg), MIN(score), MAX(score) FROM {full_table_name}")
print(f'Score stats: {result.data_array if result else err}')

# Check columns
table_info = w.tables.get(full_name=full_table_name)
print(f'\nTable: {table_info.full_name}')
print(f'Type: {table_info.table_type}')
print('Columns:')
for col in (table_info.columns or []):
    print(f'  {col.name}: {col.type_text}')

print('\n=== Synced Table Status ===')
try:
    synced = w.database.get_synced_database_table(name=synced_table_name)
    print(f'Name: {synced.name}')
    print(f'Effective DB instance: {synced.effective_database_instance_name}')
    print(f'Effective logical DB: {synced.effective_logical_database_name}')
    print(f'Unity Catalog provisioning: {synced.unity_catalog_provisioning_state}')
    if synced.data_synchronization_status:
        s = synced.data_synchronization_status
        print(f'Detailed state: {s.detailed_state}')
        print(f'Message: {s.message}')
except Exception as e:
    print(f'Error: {e}')

print('\n=== Summary ===')
print(f'Feature table: {full_table_name}')
print(f'Synced online table: {synced_table_name}')
print('All verifications complete.')
