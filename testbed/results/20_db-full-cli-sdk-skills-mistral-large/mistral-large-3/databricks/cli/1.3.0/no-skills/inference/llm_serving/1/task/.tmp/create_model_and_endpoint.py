#!/usr/bin/env python3
"""
Script to create a model in Unity Catalog and deploy it as a serving endpoint.
"""
import os
import json
import mlflow
from mlflow.pyfunc import PythonModel
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedModelInput,
    ServedModelInputWorkloadSize
)

# Add the data directory to Python path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "data"))
from scorer import score

# Set the tracking URI to use Databricks
mlflow.set_registry_uri("databricks")

# Create a wrapper class for the scorer function
class ScorerModel(PythonModel):
    def load_context(self, context):
        pass
    
    def predict(self, context, model_input):
        return score(model_input)

# Configuration
endpoint_name = "scorerb7352a"
schema_name = os.environ["MLPAB_DATABRICKS_SCHEMA"]
# Extract just the schema part (after the dot)
schema_part = schema_name.split(".")[1]
model_name = f"{schema_part}.scorer_model"

# Initialize the Workspace client
w = WorkspaceClient()

# First, register the model
print(f"Registering model {model_name}...")
with mlflow.start_run():
    # Log the model
    mlflow.pyfunc.log_model(
        artifact_path="scorer_model",
        python_model=ScorerModel(),
        registered_model_name=model_name
    )

print(f"Model {model_name} registered successfully.")

# Now create the serving endpoint
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

# Create the endpoint and wait for it to be ready
w.serving_endpoints.create_and_wait(
    name=endpoint_name,
    config=endpoint_config
)

print(f"Endpoint {endpoint_name} created successfully.")

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
    response = w.serving_endpoints.query(
        name=endpoint_name,
        data=json.dumps({"data": payload})
    )
    responses.append(response.predictions[0])
    print(f"Response: {response.predictions[0]}")

# Write the responses to the submission file
submission = {
    "endpoint_name": endpoint_name,
    "responses": responses
}

with open("submission/answers.json", "w") as f:
    json.dump(submission, f, indent=2)

print("Deployment and invocation completed successfully!")