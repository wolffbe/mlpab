#!/usr/bin/env python3
"""Debug publish_table API call."""

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

# Inspect what the API sends - look at the request body
spec = PublishSpec(
    online_store=ONLINE_STORE_NAME,
    online_table_name=TABLE_NAME,
    publish_mode=PublishSpecPublishMode.TRIGGERED,
)
print("PublishSpec as dict:", json.dumps(spec.as_dict(), indent=2))

# Try the raw API call
print("\nTrying raw API call...")
body = {
    "source_table_name": FULL_TABLE,
    "publish_spec": spec.as_dict(),
}
print("Request body:", json.dumps(body, indent=2))

try:
    result = w.api_client.do(
        "POST",
        "/api/2.0/feature-store/publish-table",
        body=body,
    )
    print("Result:", result)
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Also try the endpoint that the SDK actually uses
print("\nLooking at feature store API internals...")
import inspect
src = inspect.getsource(w.feature_store.publish_table)
print(src)
