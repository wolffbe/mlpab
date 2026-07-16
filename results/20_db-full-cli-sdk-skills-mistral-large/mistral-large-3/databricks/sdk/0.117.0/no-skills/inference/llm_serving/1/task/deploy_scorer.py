"""Deploy the scorer as a real-time endpoint on Databricks."""
import os
import json
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    ServedModelInput,
    AutoCaptureConfigInput,
    EndpointCoreConfigInput,
    ServingEndpointDetailed,
)

# Load payloads
with open('data/payloads.json', 'r') as f:
    payloads = json.load(f)

# Initialize WorkspaceClient
w = WorkspaceClient()

# Environment variables
schema = os.environ['MLPAB_DATABRICKS_SCHEMA']
catalog, schema_name = schema.split('.')
prefix = os.environ['MLPAB_DATABRICKS_PREFIX']
endpoint_name = f"{prefix}_scorerb7352a"
model_name = f"{prefix}_scorer_model"

# Register the model in Unity Catalog
full_model_name = f"{catalog}.{schema_name}.{model_name}"
model = w.model_registry.create_model(
    name=model_name,
    description="Scorer model for real-time endpoint",
)

# Create a model version from the scorer.py file
model_version = w.model_registry.create_model_version(
    name=full_model_name,
    source="file://data/scorer.py",
    run_id=None,
)

# Wait for model version to be ready
while True:
    mv = w.model_registry.get_model_version(name=full_model_name, version=model_version)
    if mv.status == "READY":
        break
    time.sleep(5)

model_version = model_version.version

# Create or update the serving endpoint
try:
    endpoint = w.serving_endpoints.get(endpoint_name)
    print(f"Endpoint {endpoint_name} already exists. Updating...")
    
    # Update the endpoint configuration
    config = EndpointCoreConfigInput(
        name=endpoint_name,
        served_models=[
            ServedModelInput(
                model_name=full_model_name,
                model_version=model_version.version,
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
                model_name=full_model_name,
                model_version=model_version.version,
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