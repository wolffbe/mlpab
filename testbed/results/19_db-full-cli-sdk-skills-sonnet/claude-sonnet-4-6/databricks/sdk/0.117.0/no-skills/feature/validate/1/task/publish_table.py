#!/usr/bin/env python3
"""Publish feature table to online store."""

import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.ml import PublishSpec, PublishSpecPublishMode

SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]
CATALOG, SCHEMA_NAME = SCHEMA.split(".", 1)
TABLE_NAME = "eventsd693d3"
FULL_TABLE = f"{CATALOG}.{SCHEMA_NAME}.{TABLE_NAME}"
ONLINE_STORE_NAME = PREFIX.replace("_", "-") + "-os"

w = WorkspaceClient()

# Verify online store exists
print("Online stores:")
stores = list(w.feature_store.list_online_stores())
for s in stores:
    print(f"  {s.name}: {s.state}")

# Try different online_table_name formats
print(f"\nPublishing {FULL_TABLE} to store {ONLINE_STORE_NAME}")

# Try with just a simple name
for name_attempt in [
    TABLE_NAME,
    f"{SCHEMA_NAME}_{TABLE_NAME}",
    f"{TABLE_NAME}_online",
]:
    print(f"\nAttempting online_table_name='{name_attempt}'...")
    try:
        result = w.feature_store.publish_table(
            source_table_name=FULL_TABLE,
            publish_spec=PublishSpec(
                online_store=ONLINE_STORE_NAME,
                online_table_name=name_attempt,
                publish_mode=PublishSpecPublishMode.TRIGGERED,
            ),
        )
        print(f"SUCCESS: {result}")
        break
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
