"""Deploy the scorer as a real-time endpoint on Databricks using Unity Catalog."""
import os
import json
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    ServedModelInput,
    EndpointCoreConfigInput,
)

# Load payloads
with open('data/payloads.json', 'r') as f:
    payloads = json.load(f)

# Initialize WorkspaceClient
w = WorkspaceClient()

# Environment variables
schema = os.environ['MLPAB_DATABRICKS_SCHEMA']
prefix = os.environ['MLPAB_DATABRICKS_PREFIX']
endpoint_name = f"{prefix}_scorerb7352a"
model_name = f"{schema}.{prefix}_scorer_model"

# Create or update the serving endpoint
try:
    endpoint = w.serving_endpoints.get(endpoint_name)
    print(f"Endpoint {endpoint_name} already exists. Updating...")
    
    # Update the endpoint configuration
    config = EndpointCoreConfigInput(
        name=endpoint_name,
        served_models=[
            ServedModelInput(
                model_name=model_name,
                model_version="1",  # Unity Catalog models start at version 1
                workload_size="Small",
                scale_to_zero_enabled=True,
            )
        ],
    )
    w.serving_endpoints.update_config_and_wait(name=endpoint_name, config=config)

except Exception:
    print(f"Creating endpoint {endpoint_name}...")
    config = EndpointCoreConfigInput(
        name=endpoint_name,
        served_models=[
            ServedModelInput(
                model_name=model_name,
                model_version="1",  # Unity Catalog models start at version 1
                workload_size="Small",
                scale_to_zero_enabled=True,
            )
        ],
    )
    endpoint = w.serving_endpoints.create_and_wait(name=endpoint_name, config=config)

# Invoke the endpoint for each payload
responses = []
for payload in payloads:
    response = w.serving_endpoints.query(
        name=endpoint_name,
        data_json={"dataframe_split": {"data": [payload]}},
    )
    responses.append(response.predictions[0]["score"])

# Write the results
with open('submission/answers.json', 'w') as f:
    json.dump({
        "endpoint_name": endpoint_name,
        "responses": responses,
    }, f)

print("Deployment and inference completed successfully.")