from databricks.sdk import WorkspaceClient
import os, time
w = WorkspaceClient()
schema = os.environ['MLPAB_DATABRICKS_SCHEMA']
catalog, schema_name = schema.split('.')

synced_table_id = f'{catalog}.{schema_name}.featuresb1ea93_online'

# Try correct path format
for fmt in [
    f'synced_tables/{synced_table_id}',
    f'synced-tables/{synced_table_id}',
    f'syncedTables/{synced_table_id}',
]:
    try:
        st_info = w.postgres.get_synced_table(name=fmt)
        print(f"Found with format '{fmt}': status={st_info.status}")
        break
    except Exception as e:
        print(f"Format '{fmt}': {str(e)[:120]}")
