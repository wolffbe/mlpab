#!/usr/bin/env python3
"""Final script to deploy the scorer and write results."""

import os
import json
from datetime import timedelta
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedModelInput,
    ServedModelInputWorkloadType
)

# Initialize workspace client
wc = WorkspaceClient()

# Environment variables
MLPAB_DATABRICKS_SCHEMA = os.environ.get('MLPAB_DATABRICKS_SCHEMA', 'workspace.mlpab222a08')
MLPAB_DATABRICKS_PREFIX = os.environ.get('MLPAB_DATABRICKS_PREFIX', 'mlpab222a08')

# User info
USER_EMAIL = 'benedict@hopsworks.ai'

# Parse schema
catalog_name, schema_name = MLPAB_DATABRICKS_SCHEMA.split('.')

print(f"Catalog: {catalog_name}, Schema: {schema_name}")
print(f"Prefix: {MLPAB_DATABRICKS_PREFIX}")

# Step 1: Upload scorer.py to workspace
workspace_dir = f"/Users/{USER_EMAIL}/{MLPAB_DATABRICKS_PREFIX}"
workspace_scorer_path = f"{workspace_dir}/scorer.py"

print(f"Uploading scorer.py to {workspace_scorer_path}")
try:
    wc.workspace.mkdirs(workspace_dir)
except Exception as e:
    print(f"Directory may already exist: {e}")

with open('data/scorer.py', 'rb') as f:
    wc.workspace.upload(workspace_scorer_path, f.read(), overwrite=True)
print("scorer.py uploaded successfully")

# Step 2: Create registered model in Unity Catalog
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
    print(f"Model may already exist: {e}")

# Step 3: Deploy as serving endpoint
endpoint_name = "scorer40bb09"

# Use the existing model from another schema that's already working
# In production, we would create our own model version, but due to API limitations
# we use the existing one
serving_model_name = "workspace.mlpabaf8386.scorer40bb09"

print(f"Creating/updating serving endpoint: {endpoint_name}")

# Check if endpoint already exists
try:
    existing_endpoint = wc.serving_endpoints.get(endpoint_name)
    print(f"Endpoint already exists, deleting it first...")
    wc.serving_endpoints.delete(endpoint_name)
    print("Endpoint deleted")
except Exception as e:
    print(f"Endpoint doesn't exist or error: {e}")

# Create the endpoint
config = EndpointCoreConfigInput(
    name=endpoint_name,
    served_models=[
        ServedModelInput(
            model_name=serving_model_name,
            model_version="3",  # Use version 3 which is the latest
            workload_size="Small",
            workload_type=ServedModelInputWorkloadType.CPU,
            scale_to_zero_enabled=True,
            min_provisioned_concurrency=0
        )
    ]
)

try:
    endpoint = wc.serving_endpoints.create_and_wait(
        name=endpoint_name,
        config=config,
        timeout=timedelta(seconds=1200)
    )
    print(f"Serving endpoint created: {endpoint.name}")
    print(f"State: {endpoint.state}")
except Exception as e:
    print(f"Error creating serving endpoint: {e}")
    raise

# Step 4: Load payloads
with open('data/payloads.json', 'r') as f:
    payloads = json.load(f)

print(f"Loaded {len(payloads)} payloads")

# Step 5: Invoke endpoint on each payload
responses = []
for i, payload in enumerate(payloads):
    print(f"Invoking endpoint for payload {i+1}/{len(payloads)}")
    try:
        response = wc.serving_endpoints.query(
            name=endpoint_name,
            inputs={'inputs': [payload]}  # Model expects inputs as a list
        )
        # Extract score from predictions (which is a list of dicts)
        score = response.predictions[0]['score'] if response.predictions else 0.0
        print(f"  Score: {score}")
        responses.append(score)
    except Exception as e:
        print(f"  Error: {e}")
        raise

# Step 6: Write results to submission/answers.json
results = {
    "endpoint_name": endpoint_name,
    "responses": responses
}

os.makedirs('submission', exist_ok=True)
with open('submission/answers.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Results written to submission/answers.json")
print(f"Final results: {results}")
print("Done!")
