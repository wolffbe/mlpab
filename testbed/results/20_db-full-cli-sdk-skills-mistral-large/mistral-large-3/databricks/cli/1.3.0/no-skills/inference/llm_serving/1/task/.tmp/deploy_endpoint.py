#!/usr/bin/env python3
"""
Script to deploy the scorer as a serving endpoint.
"""
import os
import json
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedModelInput,
    ServedModelInputWorkloadSize,
    ServingEndpointDetailed
)

# Initialize the Workspace client
w = WorkspaceClient()

# Configuration
endpoint_name = "scorerb7352a"
schema_name = os.environ["MLPAB_DATABRICKS_SCHEMA"]
model_name = f"{schema_name}.scorer_model"

# First, let's check if the endpoint already exists
try:
    endpoint = w.serving_endpoints.get(endpoint_name)
    print(f"Endpoint {endpoint_name} already exists.")
except Exception as e:
    if "RESOURCE_DOES_NOT_EXIST" in str(e):
        # Endpoint doesn't exist, create it
        print(f"Creating endpoint {endpoint_name}...")
        
        # Create the endpoint configuration
        endpoint_config = EndpointCoreConfigInput(
            name=endpoint_name,
            served_models=[
                ServedModelInput(
                    model_name=model_name,
                    model_version="1",
                    workload_size=ServedModelInputWorkloadSize.SMALL,
                    scale_to_zero_enabled=True
                )
            ]
        )
        
        # Create the endpoint
        w.serving_endpoints.create_and_wait(
            name=endpoint_name,
            config=endpoint_config
        )
        print(f"Endpoint {endpoint_name} created successfully.")
    else:
        print(f"Error checking endpoint: {e}")
        raise

# Now let's invoke the endpoint with our payloads
payloads = [
    "store lookup stream training batch online online embedding monitor store model",
    "pipeline inference store embedding training latency vector inference latency serving vector",
    "lookup training batch registry vector latency online",
    "schedule schedule stream drift batch pipeline training training store stream model inference",
    "stream store latency batch pipeline registry training feature store batch"
]

responses = []

for i, payload in enumerate(payloads):
    print(f"Invoking endpoint with payload {i+1}/{len(payloads)}...")
    try:
        response = w.serving_endpoints.query(
            name=endpoint_name,
            data=json.dumps({"data": payload})
        )
        responses.append(response.predictions[0])
        print(f"Response: {response.predictions[0]}")
    except Exception as e:
        print(f"Error invoking endpoint: {e}")
        raise

# Write the responses to the submission file
submission = {
    "endpoint_name": endpoint_name,
    "responses": responses
}

with open("submission/answers.json", "w") as f:
    json.dump(submission, f, indent=2)

print("Deployment and invocation completed successfully!")