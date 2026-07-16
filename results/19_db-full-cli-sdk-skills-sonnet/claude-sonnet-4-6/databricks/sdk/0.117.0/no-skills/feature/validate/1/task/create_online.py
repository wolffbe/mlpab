#!/usr/bin/env python3
"""Create online store and publish feature table."""

import os
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.ml import OnlineStore, PublishSpec, PublishSpecPublishMode

SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]   # workspace.mlpab394daa
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]    # mlpab394daa
CATALOG, SCHEMA_NAME = SCHEMA.split(".", 1)
TABLE_NAME = "eventsd693d3"
FULL_TABLE = f"{CATALOG}.{SCHEMA_NAME}.{TABLE_NAME}"
ONLINE_STORE_NAME = f"{PREFIX}_online_store"

w = WorkspaceClient()

print(f"Creating online store: {ONLINE_STORE_NAME}")
print(f"Source table: {FULL_TABLE}")

# Try creating online store with a small capacity
try:
    online_store = w.feature_store.create_online_store(
        OnlineStore(
            name=ONLINE_STORE_NAME,
            capacity="SMALL",
        )
    )
    print(f"Created online store: {online_store}")
except Exception as e:
    print(f"Error creating online store: {e}")
    # Try to get existing
    try:
        stores = list(w.feature_store.list_online_stores())
        print(f"Existing stores: {stores}")
    except Exception as e2:
        print(f"List stores error: {e2}")
