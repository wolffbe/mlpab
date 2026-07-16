import os
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    OnlineTable,
    OnlineTableSpec,
    OnlineTableSpecTriggeredSchedulingPolicy,
)

w = WorkspaceClient()
schema_fqn = os.environ['MLPAB_DATABRICKS_SCHEMA']
catalog_name, schema_name = schema_fqn.split('.')

table_name = 'transactions9dd1da'
full_table = f'{catalog_name}.{schema_name}.{table_name}'

print(f"Creating online table for: {full_table}")

online_table = OnlineTable(
    name=full_table,
    spec=OnlineTableSpec(
        source_table_full_name=full_table,
        primary_key_columns=['row_id'],
        timeseries_key='event_time',
        run_triggered=OnlineTableSpecTriggeredSchedulingPolicy(),
        perform_full_copy=True,
    ),
)

try:
    result = w.online_tables.create(online_table)
    print(f"Online table creation initiated: {result}")
    # Wait for it to become active
    print("Waiting for online table to become active...")
    final = w.online_tables.wait_get_online_table_active(
        name=full_table,
        timeout=__import__('datetime').timedelta(seconds=1200)
    )
    print(f"Online table active: {final.name}")
    print(f"  Status: {final.status}")
    print(f"  Serving URL: {final.table_serving_url}")
except Exception as e:
    print(f"Error creating online table: {e}")
    # Check current state
    try:
        existing = w.online_tables.get(full_table)
        print(f"Existing online table state: {existing}")
    except Exception as e2:
        print(f"Could not get online table: {e2}")
