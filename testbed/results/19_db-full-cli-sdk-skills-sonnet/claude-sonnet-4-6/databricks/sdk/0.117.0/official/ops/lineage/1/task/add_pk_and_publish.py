import os, time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

w = WorkspaceClient()
schema = os.environ['MLPAB_DATABRICKS_SCHEMA']
prefix = os.environ['MLPAB_DATABRICKS_PREFIX']
catalog, schema_name = schema.split('.')

WAREHOUSE_ID = '4dfab06c923fe3cc'

def run_sql(statement):
    resp = w.statement_execution.execute_statement(
        statement=statement,
        warehouse_id=WAREHOUSE_ID,
        wait_timeout='50s',
    )
    terminal = {StatementState.SUCCEEDED, StatementState.FAILED, StatementState.CANCELED, StatementState.CLOSED}
    while resp.status.state not in terminal:
        time.sleep(2)
        resp = w.statement_execution.get_statement(resp.statement_id)
    if resp.status.state != StatementState.SUCCEEDED:
        raise Exception(f'SQL failed ({resp.status.state}): {resp.status.error}')
    return resp

# Add primary key constraints to all three tables
tables = [
    ('rawad05474', 'pk_rawad05474'),
    ('rawbd05474', 'pk_rawbd05474'),
    ('derivedd05474', 'pk_derivedd05474'),
]

for table, pk_name in tables:
    print(f'Adding primary key to {table}...')
    try:
        run_sql(f'ALTER TABLE {catalog}.{schema_name}.{table} ADD CONSTRAINT {pk_name} PRIMARY KEY (row_id)')
        print(f'  Primary key added to {table}')
    except Exception as e:
        print(f'  Note: {e}')

# Now publish derivedd05474 to online feature store
from databricks.sdk.service.ml import PublishSpec, PublishSpecPublishMode

store_name = f'{prefix}-online-store'
source_table = f'{catalog}.{schema_name}.derivedd05474'
online_table_name = f'{catalog}.{schema_name}.derivedd05474_fs_online'

print(f'\nPublishing {source_table} to online store {store_name}...')
try:
    result = w.feature_store.publish_table(
        source_table_name=source_table,
        publish_spec=PublishSpec(
            online_store=store_name,
            online_table_name=online_table_name,
            publish_mode=PublishSpecPublishMode.SNAPSHOT,
        )
    )
    print(f'Published successfully: {result}')
except Exception as e:
    print(f'Error publishing: {type(e).__name__}: {e}')

# Check lineage
print('\nChecking Unity Catalog lineage...')
try:
    lineage = w.api_client.do('GET', f'/api/2.0/lineage-tracking/table-lineage',
        query={'table_name': f'{catalog}.{schema_name}.derivedd05474'})
    print(f'Lineage: {lineage}')
except Exception as e:
    print(f'Lineage check: {e}')
