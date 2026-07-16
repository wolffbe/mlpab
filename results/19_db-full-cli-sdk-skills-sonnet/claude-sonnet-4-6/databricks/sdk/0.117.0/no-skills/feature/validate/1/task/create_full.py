#!/usr/bin/env python3
"""Create online store (CU_1) and publish feature table for real-time access."""

import os
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.ml import OnlineStore, OnlineStoreConfig, PublishSpec, PublishSpecPublishMode

SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]
CATALOG, SCHEMA_NAME = SCHEMA.split(".", 1)
TABLE_NAME = "eventsd693d3"
FULL_TABLE = f"{CATALOG}.{SCHEMA_NAME}.{TABLE_NAME}"
ONLINE_STORE_NAME = PREFIX.replace("_", "-") + "-os"
ONLINE_TABLE_NAME = f"{TABLE_NAME}_ol"  # published table name

w = WorkspaceClient()

print(f"Online store name: {ONLINE_STORE_NAME}")
print(f"Source table: {FULL_TABLE}")

# Step 1: Create online store
print("\n[1] Creating online store...")
try:
    online_store = w.feature_store.create_online_store(
        OnlineStore(
            name=ONLINE_STORE_NAME,
            capacity="CU_1",
        )
    )
    print(f"    Created: {online_store}")
except Exception as e:
    print(f"    Error: {type(e).__name__}: {e}")
    # May already exist from a previous run - try to get it
    print("    Trying to list existing stores...")
    try:
        stores = list(w.feature_store.list_online_stores())
        for s in stores:
            print(f"      Existing store: {s.name}, state: {s.state}")
    except Exception as e2:
        print(f"      List error: {e2}")

# Step 2: Wait for online store to be ready
print("\n[2] Checking online store state...")
deadline = time.time() + 300
while time.time() < deadline:
    try:
        stores = list(w.feature_store.list_online_stores())
        our_store = next((s for s in stores if s.name == ONLINE_STORE_NAME), None)
        if our_store:
            print(f"    Store state: {our_store.state}")
            if str(our_store.state) == "OnlineStoreState.ACTIVE" or "ACTIVE" in str(our_store.state):
                print("    Store is ACTIVE!")
                break
            time.sleep(10)
        else:
            print("    Store not found in list yet...")
            time.sleep(5)
    except Exception as e:
        print(f"    Check error: {e}")
        break
else:
    print("    Timed out waiting for online store")

# Step 3: Publish feature table to online store
print("\n[3] Publishing feature table...")
try:
    publish_spec = PublishSpec(
        online_store=ONLINE_STORE_NAME,
        online_table_name=ONLINE_TABLE_NAME,
        publish_mode=PublishSpecPublishMode.TRIGGERED,
    )
    result = w.feature_store.publish_table(
        source_table_name=FULL_TABLE,
        publish_spec=publish_spec,
    )
    print(f"    Published: {result}")
except Exception as e:
    print(f"    Error: {type(e).__name__}: {e}")

print("\n=== Done ===")
