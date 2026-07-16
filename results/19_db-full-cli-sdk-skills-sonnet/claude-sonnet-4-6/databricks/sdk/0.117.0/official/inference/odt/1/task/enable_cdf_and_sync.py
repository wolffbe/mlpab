#!/usr/bin/env python3
"""Enable Change Data Feed and create synced table."""
import os
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.database import (
    SyncedDatabaseTable, SyncedTableSpec, SyncedTableSchedulingPolicy,
    DatabaseInstanceState
)
from databricks.sdk.service.sql import StatementState

w = WorkspaceClient()
prefix = os.environ['MLPAB_DATABRICKS_PREFIX']
schema_full = os.environ['MLPAB_DATABRICKS_SCHEMA']
catalog_name, schema_name = schema_full.split('.')
full_table_name = f'{catalog_name}.{schema_name}.scored50223c'
db_instance_name = f'{prefix}-db'

# Get the warehouse
warehouses = list(w.warehouses.list())
print(f'Warehouses: {[(wh.name, wh.state) for wh in warehouses]}')

if not warehouses:
    print('No warehouses available - using notebook job to enable CDF')
    # Fall back to notebook approach
else:
    warehouse_id = warehouses[0].id
    warehouse_name = warehouses[0].name
    print(f'Using warehouse: {warehouse_name} ({warehouse_id})')

    # Enable CDF on the table
    alter_sql = f"ALTER TABLE {full_table_name} SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')"
    print(f'Executing: {alter_sql}')

    from databricks.sdk.service.sql import ExecuteStatementRequestOnWaitTimeout
    stmt = w.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=alter_sql,
        catalog=catalog_name,
        schema=schema_name,
        wait_timeout='50s',
        on_wait_timeout=ExecuteStatementRequestOnWaitTimeout.CONTINUE
    )
    print(f'Statement ID: {stmt.statement_id}, status: {stmt.status}')

    # Wait for completion
    start = time.time()
    while stmt.status.state in [StatementState.PENDING, StatementState.RUNNING]:
        elapsed = int(time.time() - start)
        print(f'[{elapsed}s] Statement state: {stmt.status.state}')
        if elapsed > 120:
            print('Timeout waiting for SQL')
            break
        time.sleep(5)
        stmt = w.statement_execution.get_statement(statement_id=stmt.statement_id)

    print(f'Statement final state: {stmt.status.state}')
    if stmt.status.error:
        print(f'SQL error: {stmt.status.error}')

# Now create the synced table
synced_table_uc_name = f'{catalog_name}.{schema_name}.scored50223c_synced'
print(f'\nCreating synced table: {synced_table_uc_name}')
try:
    result = w.database.create_synced_database_table(
        synced_table=SyncedDatabaseTable(
            name=synced_table_uc_name,
            database_instance_name=db_instance_name,
            logical_database_name=schema_name,
            spec=SyncedTableSpec(
                source_table_full_name=full_table_name,
                primary_key_columns=['request_id'],
                create_database_objects_if_missing=True,
                scheduling_policy=SyncedTableSchedulingPolicy.TRIGGERED
            )
        )
    )
    print(f'Synced table created: {result}')

    # Wait for synced table to provision
    start = time.time()
    while True:
        synced = w.database.get_synced_database_table(name=synced_table_uc_name)
        elapsed = int(time.time() - start)
        prov_state = synced.unity_catalog_provisioning_state
        prov_val = prov_state.value if prov_state else 'UNKNOWN'
        sync_status = synced.data_synchronization_status
        print(f'[{elapsed}s] Synced table state: {prov_val}, sync: {sync_status}')
        if prov_val in ['ACTIVE', 'FAILED']:
            break
        if elapsed > 300:
            print('Timeout waiting for synced table')
            break
        time.sleep(20)

    print(f'Synced table final state: {prov_val}')

except Exception as e:
    print(f'Error creating synced table: {e}')

print('\nDone.')
