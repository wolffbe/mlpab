import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.postgres import (
    SyncedTable,
    SyncedTableSyncedTableSpec,
    SyncedTableSyncedTableSpecSyncedTableSchedulingPolicy,
    NewPipelineSpec,
)

w = WorkspaceClient()
prefix = os.environ['MLPAB_DATABRICKS_PREFIX']
schema = os.environ['MLPAB_DATABRICKS_SCHEMA']
catalog_name, schema_name = schema.split('.')

source_table = f'{schema}.predictions7b586d'
dest_catalog = f'{prefix}_online_cat'
synced_table_id = f'{dest_catalog}.pred_online_db.predictions7b586d'

print(f'Creating postgres synced table...')
print(f'Source: {source_table}')
print(f'Destination ID: {synced_table_id}')

try:
    op = w.postgres.create_synced_table(
        synced_table=SyncedTable(
            spec=SyncedTableSyncedTableSpec(
                source_table_full_name=source_table,
                primary_key_columns=['row_id'],
                scheduling_policy=SyncedTableSyncedTableSpecSyncedTableSchedulingPolicy.SNAPSHOT,
                create_database_objects_if_missing=True,
                postgres_database='pred_online_db',
                new_pipeline_spec=NewPipelineSpec(
                    storage_catalog=catalog_name,
                    storage_schema=schema_name,
                ),
            ),
        ),
        synced_table_id=synced_table_id,
    )
    print('Operation result:', op)
    if op:
        print('Op name:', op.operation.name if op.operation else None)
except Exception as e:
    print('Error:', str(e)[:500])
    import traceback
    traceback.print_exc()
