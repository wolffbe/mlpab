import os
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.ml import OnlineStore, PublishSpec, PublishSpecPublishMode

w = WorkspaceClient()
schema_fqn = os.environ['MLPAB_DATABRICKS_SCHEMA']
catalog_name, schema_name = schema_fqn.split('.')
prefix = os.environ['MLPAB_DATABRICKS_PREFIX']

table_name = 'transactions9dd1da'
full_table = f'{catalog_name}.{schema_name}.{table_name}'
online_store_name = f'{prefix}-fs-online'
online_table_name = f'{catalog_name}.{schema_name}.{table_name}_online'

print(f"Source table: {full_table}")
print(f"Online store name: {online_store_name}")

# Step 1: Get/Create an online store
print("\nChecking/creating online store...")
try:
    store = w.feature_store.get_online_store(online_store_name)
    print(f"Found existing online store: {store.name}, state={store.state}")
except Exception:
    try:
        store = w.feature_store.create_online_store(
            OnlineStore(
                name=online_store_name,
                capacity='CU_1',
            )
        )
        print(f"Created online store: {store.name}, state={store.state}")
    except Exception as e:
        print(f"Error creating online store: {e}")
        raise

# Wait for the online store to be available
print("Waiting for online store to become available...")
max_wait = 600
waited = 0
while waited < max_wait:
    try:
        store = w.feature_store.get_online_store(online_store_name)
        print(f"  Store state: {store.state}")
        state_str = str(store.state)
        if 'AVAILABLE' in state_str or 'ACTIVE' in state_str:
            print("Online store is available!")
            break
    except Exception as e:
        print(f"  Error checking store state: {e}")
    time.sleep(10)
    waited += 10

# Step 2: Publish the table to the online store
print(f"\nPublishing table {full_table} to online store {online_store_name}...")
try:
    pub_resp = w.feature_store.publish_table(
        source_table_name=full_table,
        publish_spec=PublishSpec(
            online_store=online_store_name,
            online_table_name=online_table_name,
            publish_mode=PublishSpecPublishMode.SNAPSHOT,
        )
    )
    print(f"Publish response: {pub_resp}")
except Exception as e:
    print(f"Error publishing table: {e}")
    import traceback
    traceback.print_exc()
