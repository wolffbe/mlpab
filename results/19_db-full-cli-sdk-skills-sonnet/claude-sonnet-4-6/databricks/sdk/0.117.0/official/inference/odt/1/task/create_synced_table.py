#!/usr/bin/env python3
"""Create a synced database table for low-latency access."""
import os
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.database import (
    DatabaseInstance, SyncedDatabaseTable, SyncedTableSpec,
    SyncedTableSchedulingPolicy
)

w = WorkspaceClient()
prefix = os.environ['MLPAB_DATABRICKS_PREFIX']
schema_full = os.environ['MLPAB_DATABRICKS_SCHEMA']
catalog_name, schema_name = schema_full.split('.')
full_table_name = f'{catalog_name}.{schema_name}.scored50223c'
db_instance_name = f'{prefix}-db'

print(f'Creating database instance: {db_instance_name}')

# Create Lakebase database instance
try:
    db_instance = w.database.create_database_instance_and_wait(
        database_instance=DatabaseInstance(
            name=db_instance_name,
            capacity='CU_1'
        ),
        timeout=__import__('datetime').timedelta(seconds=300)
    )
    print(f'Database instance created: {db_instance.name}, state: {db_instance.state}')
except Exception as e:
    print(f'Error creating instance: {e}')
    # Try to get existing
    try:
        db_instance = w.database.get_database_instance(name=db_instance_name)
        print(f'Got existing instance: {db_instance.name}, state: {db_instance.state}')
    except Exception as e2:
        print(f'Error getting existing: {e2}')
        exit(1)

# Wait for instance to be available
start = time.time()
while True:
    inst = w.database.get_database_instance(name=db_instance_name)
    elapsed = int(time.time() - start)
    state_val = inst.state.value if inst.state else 'UNKNOWN'
    print(f'[{elapsed}s] DB instance state: {state_val}')
    if state_val == 'AVAILABLE':
        break
    if elapsed > 300:
        print('Timeout waiting for database instance')
        break
    time.sleep(15)

print(f'Database instance ready: {state_val}')

# Create synced table
synced_table_name = f'{prefix}_scored50223c'
print(f'\nCreating synced table: {synced_table_name}')
try:
    result = w.database.create_synced_database_table(
        synced_table=SyncedDatabaseTable(
            name=synced_table_name,
            database_instance_name=db_instance_name,
            spec=SyncedTableSpec(
                source_table_full_name=full_table_name,
                primary_key_columns=['request_id'],
                create_database_objects_if_missing=True,
                scheduling_policy=SyncedTableSchedulingPolicy.TRIGGERED
            )
        )
    )
    print(f'Synced table created: {result}')
except Exception as e:
    print(f'Error creating synced table: {e}')
    # Check what format name should be
    try:
        help_info = dir(SyncedDatabaseTable)
        print(f'SyncedDatabaseTable fields: {help_info}')
    except Exception:
        pass

print('\nDone.')
