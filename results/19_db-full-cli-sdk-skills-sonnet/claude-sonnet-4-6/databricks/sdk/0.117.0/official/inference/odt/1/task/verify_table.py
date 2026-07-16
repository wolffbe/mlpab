#!/usr/bin/env python3
"""Verify the feature table data."""
import os
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

w = WorkspaceClient()
schema_full = os.environ['MLPAB_DATABRICKS_SCHEMA']
catalog_name, schema_name = schema_full.split('.')
full_table_name = f'{catalog_name}.{schema_name}.scored50223c'
synced_table_name = f'{catalog_name}.{schema_name}.scored50223c_synced'

# Get warehouse
warehouses = list(w.warehouses.list())
warehouse_id = warehouses[0].id

def run_sql(sql):
    from databricks.sdk.service.sql import ExecuteStatementRequestOnWaitTimeout
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

# Check row count
result, err = run_sql(f"SELECT COUNT(*) as cnt FROM {full_table_name}")
if err:
    print(f'Count error: {err}')
else:
    print(f'Row count: {result.data_array}')

# Check sample data
result, err = run_sql(f"SELECT * FROM {full_table_name} ORDER BY request_id LIMIT 5")
if err:
    print(f'Sample error: {err}')
else:
    print(f'Columns: {result.chunk and "data" or "no chunk"}')
    if result.data_array:
        for row in result.data_array:
            print(f'  {row}')

# Check distinct request_ids
result, err = run_sql(f"SELECT COUNT(DISTINCT request_id) as dist_count FROM {full_table_name}")
if err:
    print(f'Distinct count error: {err}')
else:
    print(f'Distinct request_ids: {result.data_array}')

# Check the synced table status
try:
    synced = w.database.get_synced_database_table(name=synced_table_name)
    print(f'\nSynced table: {synced_table_name}')
    print(f'  Effective DB instance: {synced.effective_database_instance_name}')
    print(f'  Effective logical DB: {synced.effective_logical_database_name}')
    if synced.data_synchronization_status:
        s = synced.data_synchronization_status
        print(f'  Detailed state: {s.detailed_state}')
        print(f'  Message: {s.message}')
        if s.triggered_update_status:
            print(f'  Triggered update: {s.triggered_update_status}')
except Exception as e:
    print(f'Error getting synced table: {e}')

# Show table properties
table_info = w.tables.get(full_name=full_table_name)
print(f'\nTable type: {table_info.table_type}')
print(f'Properties with delta: {dict((k,v) for k,v in (table_info.properties or {}).items() if "change" in k.lower() or "cdf" in k.lower())}')
