#!/usr/bin/env python3
"""Create online store with valid DNS name and publish feature table."""

import os
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.ml import OnlineStore, OnlineStoreConfig, PublishSpec, PublishSpecPublishMode

SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]
CATALOG, SCHEMA_NAME = SCHEMA.split(".", 1)
TABLE_NAME = "eventsd693d3"
FULL_TABLE = f"{CATALOG}.{SCHEMA_NAME}.{TABLE_NAME}"
# Must be DNS-compliant: alphanumeric + hyphens
ONLINE_STORE_NAME = PREFIX.replace("_", "-") + "-os"

w = WorkspaceClient()

print(f"Online store name: {ONLINE_STORE_NAME}")
print(f"Source table: {FULL_TABLE}")

# Try creating online store
try:
    online_store = w.feature_store.create_online_store(
        OnlineStore(
            name=ONLINE_STORE_NAME,
            capacity="SMALL",
        )
    )
    print(f"Created online store: {online_store}")
except Exception as e:
    print(f"Error creating online store: {type(e).__name__}: {e}")

# Try listing stores to see what exists
print("\nListing online stores:")
try:
    stores = list(w.feature_store.list_online_stores())
    for s in stores:
        print(f"  Store: {s}")
    if not stores:
        print("  No stores found")
except Exception as e:
    print(f"Error: {e}")
