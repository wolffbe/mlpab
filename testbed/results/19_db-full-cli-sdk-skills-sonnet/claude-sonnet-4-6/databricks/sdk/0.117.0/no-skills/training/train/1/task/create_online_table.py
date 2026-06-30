import os
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    OnlineTable,
    OnlineTableSpec,
    OnlineTableSpecTriggeredSchedulingPolicy,
)

w = WorkspaceClient()
schema = os.environ['MLPAB_DATABRICKS_SCHEMA']

online_table_name = f'{schema}.predictions7b586d'
source_table_name = f'{schema}.predictions7b586d'

print('Creating online table for:', source_table_name)

online_table = OnlineTable(
    name=online_table_name,
    spec=OnlineTableSpec(
        source_table_full_name=source_table_name,
        primary_key_columns=['row_id'],
        run_triggered=OnlineTableSpecTriggeredSchedulingPolicy(),
        perform_full_copy=True,
    )
)

try:
    result = w.online_tables.create_and_wait(table=online_table)
    print('Online table created:', result.name)
    print('Status:', result.status)
except Exception as e:
    print('Error creating online table:', e)
    # Try getting it in case it exists
    try:
        existing = w.online_tables.get(name=online_table_name)
        print('Existing online table:', existing.name)
        print('Status:', existing.status)
    except Exception as e2:
        print('Get error:', e2)
