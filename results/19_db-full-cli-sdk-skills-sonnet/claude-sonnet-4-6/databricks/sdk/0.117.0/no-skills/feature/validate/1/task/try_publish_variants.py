#!/usr/bin/env python3
"""Try different publish_spec formats."""

import os
import json
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.ml import PublishSpec, PublishSpecPublishMode

SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]
CATALOG, SCHEMA_NAME = SCHEMA.split(".", 1)
TABLE_NAME = "eventsd693d3"
FULL_TABLE = f"{CATALOG}.{SCHEMA_NAME}.{TABLE_NAME}"
ONLINE_STORE_NAME = PREFIX.replace("_", "-") + "-os"

w = WorkspaceClient()

# Try with full 3-part name for online_table_name
attempts = [
    {"online_table_name": f"{CATALOG}.{SCHEMA_NAME}.{TABLE_NAME}_online"},
    {"online_table_name": f"{SCHEMA_NAME}.{TABLE_NAME}_online"},
    {"online_table_name": TABLE_NAME + "_online"},
]

for attempt in attempts:
    spec_dict = {
        "online_store": ONLINE_STORE_NAME,
        "publish_mode": "TRIGGERED",
        **attempt,
    }
    print(f"\nAttempting: {json.dumps(spec_dict, indent=2)}")
    try:
        result = w.api_client.do(
            "POST",
            f"/api/2.0/feature-store/tables/{FULL_TABLE}/publish",
            body={"publish_spec": spec_dict},
            headers={"Accept": "application/json", "Content-Type": "application/json"},
        )
        print(f"SUCCESS: {result}")
        break
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
