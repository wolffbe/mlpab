import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.database import (
    DatabaseCatalog,
    SyncedDatabaseTable,
    SyncedTableSpec,
    SyncedTableSchedulingPolicy,
    NewPipelineSpec,
)

w = WorkspaceClient()
prefix = os.environ['MLPAB_DATABRICKS_PREFIX']
schema = os.environ['MLPAB_DATABRICKS_SCHEMA']
catalog_name, schema_name = schema.split('.')
instance_name = f'{prefix}-online-store'
source_table = f'{schema}.predictions7b586d'

print('Creating database catalog...')
try:
    db_catalog = w.database.create_database_catalog(
        catalog=DatabaseCatalog(
            name=f'{prefix}-predictions-catalog',
            database_instance_name=instance_name,
            database_name='predictions_db',
            create_database_if_not_exists=True,
        )
    )
    print('Database catalog created:', db_catalog.name)
except Exception as e:
    print('Catalog creation error (may already exist):', e)

print()
print('Creating synced database table...')
print('Source table:', source_table)

try:
    synced_table = w.database.create_synced_database_table(
        synced_table=SyncedDatabaseTable(
            name=f'{prefix}-predictions-synced',
            database_instance_name=instance_name,
            spec=SyncedTableSpec(
                source_table_full_name=source_table,
                primary_key_columns=['row_id'],
                scheduling_policy=SyncedTableSchedulingPolicy.TRIGGERED,
                new_pipeline_spec=NewPipelineSpec(
                    storage_catalog=catalog_name,
                    storage_schema=schema_name,
                ),
            ),
        )
    )
    print('Synced table created:', synced_table.name)
    print('Status:', synced_table.data_synchronization_status)
    print('Spec:', synced_table.spec)
except Exception as e:
    print('Synced table creation error:', e)
    import traceback
    traceback.print_exc()
