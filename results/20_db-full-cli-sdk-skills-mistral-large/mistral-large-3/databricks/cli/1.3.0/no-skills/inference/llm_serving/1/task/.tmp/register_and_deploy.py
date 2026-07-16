# Databricks notebook source
# MAGIC %md
# MAGIC ## Register and Deploy Scorer Model

# COMMAND ----------

import os
import json
import mlflow
from mlflow.pyfunc import PythonModel

# Add the scorer.py file to the path
import sys
sys.path.append("/dbfs/FileStore/{}/".format(os.environ["MLPAB_DATABRICKS_PREFIX"]))
from scorer import score

# Create a wrapper class for the scorer function
class ScorerModel(PythonModel):
    def load_context(self, context):
        pass
    
    def predict(self, context, model_input):
        return score(model_input)

# Set the tracking URI to use Databricks
mlflow.set_registry_uri("databricks")

# Register the model
model_name = "{}.scorer_model".format(os.environ["MLPAB_DATABRICKS_SCHEMA"])

with mlflow.start_run():
    mlflow.pyfunc.log_model(
        artifact_path="scorer_model",
        python_model=ScorerModel(),
        registered_model_name=model_name
    )

print(f"Model registered as: {model_name}")

# COMMAND ----------

# Now create the serving endpoint
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

# Write the responses to DBFS
result = {
    "endpoint_name": endpoint_name,
    "responses": responses
}

with open(f"/dbfs/FileStore/{os.environ['MLPAB_DATABRICKS_PREFIX']}/answers.json", "w") as f:
    json.dump(result, f)

print("Deployment and invocation completed successfully!")
print(f"Responses: {responses}")