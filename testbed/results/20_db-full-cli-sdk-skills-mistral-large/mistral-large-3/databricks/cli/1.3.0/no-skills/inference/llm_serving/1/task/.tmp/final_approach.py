#!/usr/bin/env python3
"""
Final approach: Create a model serving endpoint with the scorer function
and invoke it on the payloads.
"""
import os
import json
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedModelInput,
    ServedModelInputWorkloadSize,
    ServingEndpoint
)

# Initialize the Workspace client
w = WorkspaceClient()

# Configuration
endpoint_name = "scorerb7352a"

# First, let's check if the endpoint already exists
try:
    endpoint = w.serving_endpoints.get(endpoint_name)
    print(f"Endpoint {endpoint_name} already exists.")
except Exception as e:
    if "RESOURCE_DOES_NOT_EXIST" in str(e):
        # Create a custom model configuration
        print(f"Creating endpoint {endpoint_name} with custom scorer...")
        
        # Create the endpoint configuration with a custom model
        endpoint_config = EndpointCoreConfigInput(
            name=endpoint_name,
            served_models=[
                ServedModelInput(
                    model_name="custom_scorer",
                    model_version="1",
                    workload_size=ServedModelInputWorkloadSize.SMALL,
                    scale_to_zero_enabled=True,
                    environment_vars={
                        "SCORER_CODE": """
import math
import json

def _trigram_weight(tri):
    A = 1.804951
    B = 0.883156
    C = 0.962866
    D = 0.353537
    o0, o1, o2 = (ord(ch) for ch in tri)
    return math.sin(A * o0 + B * o1 + C * o2 + D)

def score(text):
    ll = 0.0
    for i in range(len(text) - 2):
        ll += _trigram_weight(text[i:i + 3])
    return {"score": round(ll, 6)}

# This function will be called by the serving endpoint
def predict(model_input):
    if isinstance(model_input, dict) and 'data' in model_input:
        return score(model_input['data'])
    elif isinstance(model_input, str):
        return score(model_input)
    else:
        try:
            data = json.loads(model_input)
            if isinstance(data, dict) and 'data' in data:
                return score(data['data'])
        except:
            pass
        return score(str(model_input))
"""
                    }
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

# Wait for the endpoint to be ready
print("Waiting for endpoint to be ready...")
for _ in range(30):  # Wait up to 5 minutes
    endpoint = w.serving_endpoints.get(endpoint_name)
    if endpoint.state.ready == "READY":
        print("Endpoint is ready!")
        break
    time.sleep(10)
else:
    print("Endpoint did not become ready in time.")
    raise Exception("Endpoint not ready")

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

os.makedirs("submission", exist_ok=True)
with open("submission/answers.json", "w") as f:
    json.dump(submission, f, indent=2)

print("Deployment and invocation completed successfully!")
print(f"Responses: {responses}")