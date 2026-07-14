# Databricks notebook source
# MAGIC %md
# MAGIC ## Simple Scorer Deployment

# COMMAND ----------

import os
import json
import mlflow
from mlflow.pyfunc import PythonModel

# COMMAND ----------

# Define the scorer function directly
import math

def _trigram_weight(tri):
    A = 1.804951
    B = 0.883156
    C = 0.962866
    D = 0.353537
    o0, o1, o2 = (ord(ch) for ch in tri)
    return math.sin(A * o0 + B * o1 + C * o2 + D)

def score(text):
    """Log-likelihood of `text` under the trigram model."""
    ll = 0.0
    for i in range(len(text) - 2):
        ll += _trigram_weight(text[i:i + 3])
    return {"score": round(ll, 6)}

# COMMAND ----------

# Create a wrapper class for the scorer function
class ScorerModel(PythonModel):
    def load_context(self, context):
        pass
    
    def predict(self, context, model_input):
        return score(model_input)

# COMMAND ----------

# Set the tracking URI to use Databricks
mlflow.set_registry_uri("databricks")

# Register the model
schema_name = os.environ["MLPAB_DATABRICKS_SCHEMA"]
model_name = f"{schema_name}.scorer_model"

with mlflow.start_run():
    mlflow.pyfunc.log_model(
        artifact_path="scorer_model",
        python_model=ScorerModel(),
        registered_model_name=model_name
    )

print(f"Model registered as: {model_name}")

# COMMAND ----------

# Create the serving endpoint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedModelInput,
    ServedModelInputWorkloadSize
)

w = WorkspaceClient()
endpoint_name = "scorerb7352a"

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

# COMMAND ----------

# Now invoke the endpoint with our payloads
payloads = [
    "store lookup stream training batch online online embedding monitor store model",
    "pipeline inference store embedding training latency vector inference latency serving vector",
    "lookup training batch registry vector latency online",
    "schedule schedule stream drift batch pipeline training training store stream model inference",
    "stream store latency batch pipeline registry training feature store batch"
]

responses = []

for payload in payloads:
    response = w.serving_endpoints.query(
        name=endpoint_name,
        data=json.dumps({"data": payload})
    )
    responses.append(response.predictions[0])

# Write the responses to the submission file
result = {
    "endpoint_name": endpoint_name,
    "responses": responses
}

# Write to DBFS
submission_path = f"/dbfs/FileStore/{os.environ['MLPAB_DATABRICKS_PREFIX']}/answers.json"
with open(submission_path, "w") as f:
    json.dump(result, f, indent=2)

# Also write to the local submission directory
os.makedirs("/databricks/driver/submission", exist_ok=True)
with open("/databricks/driver/submission/answers.json", "w") as f:
    json.dump(result, f, indent=2)

print("Deployment and invocation completed successfully!")
print(f"Responses: {responses}")