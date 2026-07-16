"""Publish feature table to online store for low-latency lookup."""
import os
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.ml import (
    OnlineStoreState, PublishSpec, PublishSpecPublishMode
)

SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]
TABLE_NAME = "scaled7ecfaf"
FULL_TABLE = f"{SCHEMA}.{TABLE_NAME}"
STORE_NAME = PREFIX + "-store"
# Online table name: use the same full table name (published table)
ONLINE_TABLE_NAME = f"{SCHEMA}.{TABLE_NAME}"

w = WorkspaceClient()

# Check online store state
print(f"Checking online store: {STORE_NAME}")
store = w.feature_store.get_online_store(name=STORE_NAME)
print(f"Store state: {store.state}")

if store.state != OnlineStoreState.AVAILABLE:
    print("Waiting for online store to be AVAILABLE...")
    deadline = time.time() + 600
    while time.time() < deadline:
        store = w.feature_store.get_online_store(name=STORE_NAME)
        print(f"  State: {store.state}")
        if store.state == OnlineStoreState.AVAILABLE:
            break
        time.sleep(15)
    else:
        raise TimeoutError("Online store not available in time")

print("Online store is AVAILABLE.")

# Publish the table
print(f"Publishing {FULL_TABLE} to online store...")
try:
    result = w.feature_store.publish_table(
        source_table_name=FULL_TABLE,
        publish_spec=PublishSpec(
            online_store=STORE_NAME,
            online_table_name=ONLINE_TABLE_NAME,
            publish_mode=PublishSpecPublishMode.TRIGGERED,
        )
    )
    print(f"Publish result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    # Try with SNAPSHOT mode
    try:
        result = w.feature_store.publish_table(
            source_table_name=FULL_TABLE,
            publish_spec=PublishSpec(
                online_store=STORE_NAME,
                online_table_name=ONLINE_TABLE_NAME,
                publish_mode=PublishSpecPublishMode.SNAPSHOT,
            )
        )
        print(f"Publish result (SNAPSHOT): {result}")
    except Exception as e2:
        print(f"SNAPSHOT error: {type(e2).__name__}: {e2}")
