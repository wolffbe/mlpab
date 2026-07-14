#!/usr/bin/env python3
"""Publish the feature table to the online store for low-latency access."""
import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.ml import PublishSpec, PublishSpecPublishMode

w = WorkspaceClient()
prefix = os.environ['MLPAB_DATABRICKS_PREFIX']
schema_full = os.environ['MLPAB_DATABRICKS_SCHEMA']
catalog_name, schema_name = schema_full.split('.')
store_name = f'{prefix}-store'
full_table_name = f'{catalog_name}.{schema_name}.scored50223c'

# Check the online store state first
store = w.feature_store.get_online_store(name=store_name)
print(f'Online store: {store.name}, state: {store.state}')

# Try different name formats
for name_format in [
    'scored50223c',
    f'{schema_name}.scored50223c',
    f'{catalog_name}.{schema_name}.scored50223c',
    f'{catalog_name}_{schema_name}_scored50223c',
]:
    print(f'Trying online_table_name={repr(name_format)}')
    try:
        result = w.feature_store.publish_table(
            source_table_name=full_table_name,
            publish_spec=PublishSpec(
                online_store=store_name,
                online_table_name=name_format,
                publish_mode=PublishSpecPublishMode.SNAPSHOT
            )
        )
        print(f'  Success: {result}')
        break
    except Exception as e:
        print(f'  Error: {e}')
