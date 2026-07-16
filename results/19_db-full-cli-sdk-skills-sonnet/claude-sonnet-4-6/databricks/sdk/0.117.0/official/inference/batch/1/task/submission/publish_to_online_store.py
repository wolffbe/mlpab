"""
Wait for the online store to be ready and publish the feature table to it.
"""
import os
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.ml import (
    OnlineStore,
    OnlineStoreState,
    PublishSpec,
    PublishSpecPublishMode,
)

SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]
SOURCE_TABLE = f"{SCHEMA}.scores4f5893"
ONLINE_STORE_NAME = f"{PREFIX}-online-store"
# The online table name - can be same schema but different name to avoid conflict
ONLINE_TABLE_NAME = f"{SCHEMA}.scores4f5893"

w = WorkspaceClient()
print(f"Connected to Databricks: {w.config.host}")
print(f"Source table: {SOURCE_TABLE}")
print(f"Online store: {ONLINE_STORE_NAME}")

# Wait for the online store to be available
print(f"\nWaiting for online store '{ONLINE_STORE_NAME}' to be available...")
max_wait = 600
start = time.time()
while time.time() - start < max_wait:
    try:
        store = w.feature_store.get_online_store(name=ONLINE_STORE_NAME)
        state = store.state
        print(f"Online store state: {state}")
        if state == OnlineStoreState.AVAILABLE:
            print("Online store is available!")
            break
        elif state in (OnlineStoreState.FAILING_OVER,):
            print("Still starting...")
    except Exception as e:
        print(f"Error getting online store status: {e}")
    time.sleep(15)

# Check final state
store = w.feature_store.get_online_store(name=ONLINE_STORE_NAME)
print(f"Online store final state: {store.state}")
if store.state != OnlineStoreState.AVAILABLE:
    print(f"Warning: Online store not available yet, trying to publish anyway...")

# Publish the source table to the online store
print(f"\nPublishing {SOURCE_TABLE} to online store {ONLINE_STORE_NAME}...")
try:
    result = w.feature_store.publish_table(
        source_table_name=SOURCE_TABLE,
        publish_spec=PublishSpec(
            online_store=ONLINE_STORE_NAME,
            online_table_name=ONLINE_TABLE_NAME,
            publish_mode=PublishSpecPublishMode.SNAPSHOT,
        ),
    )
    print(f"Publish result: {result}")
except Exception as e:
    print(f"Error publishing table: {e}")
    # Try with TRIGGERED mode
    try:
        result = w.feature_store.publish_table(
            source_table_name=SOURCE_TABLE,
            publish_spec=PublishSpec(
                online_store=ONLINE_STORE_NAME,
                online_table_name=ONLINE_TABLE_NAME,
                publish_mode=PublishSpecPublishMode.TRIGGERED,
            ),
        )
        print(f"Publish result (triggered): {result}")
    except Exception as e2:
        print(f"Error publishing table (triggered): {e2}")

print("\nDone!")
