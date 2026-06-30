#!/usr/bin/env python3
"""Try direct API call to feature store publish."""

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

spec = PublishSpec(
    online_store=ONLINE_STORE_NAME,
    online_table_name=TABLE_NAME,
    publish_mode=PublishSpecPublishMode.TRIGGERED,
)

# Try the actual endpoint
endpoint = f"/api/2.0/feature-store/tables/{FULL_TABLE}/publish"
print(f"Endpoint: {endpoint}")
print(f"Body: {json.dumps({'publish_spec': spec.as_dict()}, indent=2)}")

try:
    result = w.api_client.do(
        "POST",
        endpoint,
        body={"publish_spec": spec.as_dict()},
        headers={"Accept": "application/json", "Content-Type": "application/json"},
    )
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Also try URL-encoding the dots
import urllib.parse
encoded_name = urllib.parse.quote(FULL_TABLE, safe='')
endpoint2 = f"/api/2.0/feature-store/tables/{encoded_name}/publish"
print(f"\nEncoded endpoint: {endpoint2}")
try:
    result = w.api_client.do(
        "POST",
        endpoint2,
        body={"publish_spec": spec.as_dict()},
        headers={"Accept": "application/json", "Content-Type": "application/json"},
    )
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
