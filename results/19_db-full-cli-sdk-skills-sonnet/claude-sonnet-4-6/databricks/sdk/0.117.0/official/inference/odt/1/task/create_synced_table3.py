#!/usr/bin/env python3
"""Create synced table with logical_database_name specified."""
import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.database import (
    SyncedDatabaseTable, SyncedTableSpec, SyncedTableSchedulingPolicy,
    DatabaseCatalog
)

w = WorkspaceClient()
prefix = os.environ['MLPAB_DATABRICKS_PREFIX']
schema_full = os.environ['MLPAB_DATABRICKS_SCHEMA']
catalog_name, schema_name = schema_full.split('.')
full_table_name = f'{catalog_name}.{schema_name}.scored50223c'
db_instance_name = f'{prefix}-db'

# First, check if we need to create a logical database (catalog)
print('Checking database API methods:', [m for m in dir(w.database) if not m.startswith('_')])

# Try to create a database catalog (logical database)
db_catalog_name = f'{schema_name}'
print(f'\nCreating database catalog: {db_catalog_name}')
try:
    db_cat = w.database.create_database_catalog(
        catalog=DatabaseCatalog(
            database_instance_name=db_instance_name,
            name=db_catalog_name
        )
    )
    print(f'  Created: {db_cat}')
except Exception as e:
    print(f'  Error: {e}')

# Now try creating synced table with the 3-part UC name and logical_database_name
synced_table_uc_name = f'{catalog_name}.{schema_name}.scored50223c_synced'
print(f'\nCreating synced table: {synced_table_uc_name}')

for logical_db in [db_catalog_name, 'main', 'default', db_instance_name]:
    print(f'  Trying logical_database_name={repr(logical_db)}')
    try:
        result = w.database.create_synced_database_table(
            synced_table=SyncedDatabaseTable(
                name=synced_table_uc_name,
                database_instance_name=db_instance_name,
                logical_database_name=logical_db,
                spec=SyncedTableSpec(
                    source_table_full_name=full_table_name,
                    primary_key_columns=['request_id'],
                    create_database_objects_if_missing=True,
                    scheduling_policy=SyncedTableSchedulingPolicy.TRIGGERED
                )
            )
        )
        print(f'  Success: {result}')
        break
    except Exception as e:
        print(f'  Error: {e}')

print('\nDone.')
