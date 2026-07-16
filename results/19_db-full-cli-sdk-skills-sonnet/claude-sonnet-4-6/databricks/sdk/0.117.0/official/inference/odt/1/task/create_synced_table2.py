#!/usr/bin/env python3
"""Try different name formats for synced database table."""
import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.database import (
    SyncedDatabaseTable, SyncedTableSpec, SyncedTableSchedulingPolicy
)

w = WorkspaceClient()
prefix = os.environ['MLPAB_DATABRICKS_PREFIX']
schema_full = os.environ['MLPAB_DATABRICKS_SCHEMA']
catalog_name, schema_name = schema_full.split('.')
full_table_name = f'{catalog_name}.{schema_name}.scored50223c'
db_instance_name = f'{prefix}-db'

# Try different name formats for the synced table
for name_format in [
    f'{catalog_name}.{schema_name}.scored50223c_synced',
    f'{schema_name}.scored50223c_synced',
    'public.scored50223c_synced',
    'scored50223c_synced',
]:
    print(f'Trying name={repr(name_format)}')
    try:
        result = w.database.create_synced_database_table(
            synced_table=SyncedDatabaseTable(
                name=name_format,
                database_instance_name=db_instance_name,
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
