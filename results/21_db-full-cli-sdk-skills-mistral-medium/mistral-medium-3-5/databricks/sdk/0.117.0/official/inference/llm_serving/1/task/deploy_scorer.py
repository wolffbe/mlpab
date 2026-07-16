#!/usr/bin/env python3
"""Script to deploy the scorer as a real-time endpoint and invoke it on payloads."""

import os
import json
import time
from databricks.sdk import WorkspaceClient

# Environment variables
MLPAB_DATABRICKS_SCHEMA = os.environ.get('MLPAB_DATABRICKS_SCHEMA', 'workspace.mlpab222a08')
MLPAB_DATABRICKS_PREFIX = os.environ.get('MLPAB_DATABRICKS_PREFIX', 'mlpab222a08')

# User info
USER_EMAIL = 'benedict@hopsworks.ai'

# Parse schema
catalog_name, schema_name = MLPAB_DATABRICKS_SCHEMA.split('.')

print(f"Catalog: {catalog_name}, Schema: {schema_name}")
print(f"Prefix: {MLPAB_DATABRICKS_PREFIX}")

# Initialize workspace client
wc = WorkspaceClient()

# Step 1: Upload scorer.py to workspace
workspace_path = f"/Users/{USER_EMAIL}/{MLPAB_DATABRICKS_PREFIX}/scorer.py"
print(f"Uploading scorer.py to {workspace_path}")

with open('data/scorer.py', 'rb') as f:
    wc.workspace.upload(workspace_path, f.read(), overwrite=True)

print("scorer.py uploaded successfully")

# Step 2: Create a registered model in Unity Catalog
model_name = f"{MLPAB_DATABRICKS_PREFIX}_scorer_model"
full_model_name = f"{catalog_name}.{schema_name}.{model_name}"

print(f"Creating registered model: {full_model_name}")
try:
    registered_model = wc.registered_models.create(
        catalog_name=catalog_name,
        schema_name=schema_name,
        name=model_name,
        comment="Custom scorer model for ML platform task"
    )
    print(f"Registered model created: {registered_model}")
except Exception as e:
    print(f"Error creating registered model: {e}")
    # Maybe it already exists, try to get it
    try:
        registered_model = wc.registered_models.get(f"{catalog_name}.{schema_name}.{model_name}")
        print(f"Registered model already exists: {registered_model}")
    except Exception as e2:
        print(f"Error getting registered model: {e2}")
        raise

# Step 3: Create a model version
# The source should be the workspace path
model_version_source = f"dbfs:/Users/{USER_EMAIL}/{MLPAB_DATABRICKS_PREFIX}/scorer.py"

print(f"Creating model version with source: {model_version_source}")
try:
    model_version = wc.model_registry.create_model_version(
        name=full_model_name,
        source=model_version_source,
        description="Initial version of scorer model"
    )
    print(f"Model version created: {model_version}")
except Exception as e:
    print(f"Error creating model version: {e}")
    raise

# Step 4: Deploy as serving endpoint
endpoint_name = "scorer40bb09"

print(f"Creating serving endpoint: {endpoint_name}")

# We need to use ServedModelInput or ServedEntityInput
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedModelInput,
    ServedEntityInput
)

# Try using ServedModelInput with the model name
config = EndpointCoreConfigInput(
    name=endpoint_name,
    served_models=[
        ServedModelInput(
            model_name=full_model_name,
            model_version="1",
            workload_size="SMALL",
            scale_to_zero_enabled=True,
            min_provisioned_concurrency=0
        )
    ]
)

try:
    endpoint = wc.serving_endpoints.create_and_wait(
        name=endpoint_name,
        config=config,
        timeout=1200  # 20 minutes timeout
    )
    print(f"Serving endpoint created: {endpoint}")
except Exception as e:
    print(f"Error creating serving endpoint: {e}")
    raise

# Step 5: Load payloads
with open('data/payloads.json', 'r') as f:
    payloads = json.load(f)

print(f"Loaded {len(payloads)} payloads")

# Step 6: Invoke endpoint on each payload
responses = []
for i, payload in enumerate(payloads):
    print(f"Invoking endpoint for payload {i+1}/{len(payloads)}")
    try:
        response = wc.serving_endpoints.query(
            name=endpoint_name,
            inputs={"text": payload}
        )
        print(f"Response: {response}")
        responses.append(response)
    except Exception as e:
        print(f"Error invoking endpoint: {e}")
        raise

# Step 7: Write results to submission/answers.json
results = {
    "endpoint_name": endpoint_name,
    "responses": responses
}

os.makedirs('submission', exist_ok=True)
with open('submission/answers.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Results written to submission/answers.json")
print("Done!")
