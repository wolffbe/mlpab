import os
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.ml import (
    OnlineStore,
    OnlineStoreConfig,
    PublishSpec,
    PublishSpecPublishMode,
    OnlineStoreState,
)

w = WorkspaceClient()
prefix = os.environ['MLPAB_DATABRICKS_PREFIX']
schema = os.environ['MLPAB_DATABRICKS_SCHEMA']
catalog_name, schema_name = schema.split('.')

online_store_name = f'{prefix}-online-store'
source_table = f'{schema}.predictions7b586d'

print(f'Creating online store: {online_store_name}')

# Create online store
try:
    store = w.feature_store.create_online_store(
        online_store=OnlineStore(
            name=online_store_name,
            capacity='CU_1',
        )
    )
    print('Online store created:', store.name, 'state:', store.state)
except Exception as e:
    print('Create error:', e)
    # Check if it already exists
    try:
        store = w.feature_store.get_online_store(name=online_store_name)
        print('Existing store:', store.name, 'state:', store.state)
    except Exception as e2:
        print('Get error:', e2)
        store = None

if store:
    # Wait for store to be available
    print('Waiting for online store to be available...')
    max_wait = 300
    start = time.time()
    while time.time() - start < max_wait:
        try:
            store = w.feature_store.get_online_store(name=online_store_name)
            print(f'  State: {store.state}')
            if store.state == OnlineStoreState.AVAILABLE:
                print('Online store is available!')
                break
            elif store.state in (OnlineStoreState.DELETING, OnlineStoreState.STOPPED):
                print('Store in unexpected state:', store.state)
                break
        except Exception as e:
            print('  Wait error:', e)
        time.sleep(15)
    else:
        print('Timeout waiting for online store')

    # Publish the table to online store
    print(f'Publishing {source_table} to online store...')
    try:
        result = w.feature_store.publish_table(
            source_table_name=source_table,
            publish_spec=PublishSpec(
                online_store=online_store_name,
                online_table_name='predictions7b586d',
                publish_mode=PublishSpecPublishMode.TRIGGERED,
            )
        )
        print('Publish result:', result)
    except Exception as e:
        print('Publish error:', e)

print('Done!')
