"""Deploy scorer.py as a Databricks Model Serving endpoint.

Strategy:
1. Run a serverless notebook job with MLflow in the environment dependencies
2. The notebook logs the scorer model to MLflow and registers it in UC
3. Create Model Serving endpoint pointing to the registered model
4. Invoke endpoint on payloads
"""
import os
import io
import json
import base64
import datetime
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs, serving
from databricks.sdk.service.workspace import Language, ImportFormat
from databricks.sdk.service import compute

# Config
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]
SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]  # workspace.mlpabbc835d
ENDPOINT_BASE = "scorer98c848"
ENDPOINT_NAME = f"{PREFIX}_{ENDPOINT_BASE}"
USER = "benedict@logicalclocks.com"
NOTEBOOK_PATH = f"/Users/{USER}/{PREFIX}/scorer_deploy_v2"

# Parse catalog and schema
catalog, schema_name = SCHEMA.split(".")  # workspace, mlpabbc835d
MODEL_NAME = f"{catalog}.{schema_name}.scorer98c848"

print(f"Endpoint name: {ENDPOINT_NAME}")
print(f"Model name: {MODEL_NAME}")
print(f"Notebook path: {NOTEBOOK_PATH}")

w = WorkspaceClient()

# Create notebook directory
try:
    w.workspace.mkdirs(f"/Users/{USER}/{PREFIX}")
except Exception:
    pass

# Notebook code that registers the scorer via MLflow (without %pip magic)
# Uses mlflow directly (already in serverless environment after dependency install)
notebook_code = f"""# Databricks notebook source
import mlflow
import mlflow.pyfunc
import json
import math

# Model constants
A = 2.158311
B = 2.825397
C = 1.859128
D = -0.317766

def _trigram_weight(tri):
    o0, o1, o2 = (ord(ch) for ch in tri)
    return math.sin(A * o0 + B * o1 + C * o2 + D)

class ScorerModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params=None):
        import math
        A, B, C, D = 2.158311, 2.825397, 1.859128, -0.317766
        def _tw(tri):
            o0, o1, o2 = (ord(ch) for ch in tri)
            return math.sin(A * o0 + B * o1 + C * o2 + D)
        if hasattr(model_input, 'iloc'):
            texts = model_input.iloc[:, 0].tolist()
        elif isinstance(model_input, dict):
            vals = list(model_input.values())
            texts = vals[0] if vals and isinstance(vals[0], list) else [str(vals[0])] if vals else []
        elif isinstance(model_input, list):
            texts = []
            for item in model_input:
                if isinstance(item, dict):
                    texts.append(item.get('text', item.get('inputs', str(item))))
                else:
                    texts.append(str(item))
        else:
            texts = [str(model_input)]
        results = []
        for text in texts:
            ll = 0.0
            for i in range(len(text) - 2):
                ll += _tw(text[i:i + 3])
            results.append({{'score': round(ll, 6)}})
        return results

MODEL_NAME = "{MODEL_NAME}"
print(f"Registering model: {{MODEL_NAME}}")

mlflow.set_registry_uri("databricks-uc")

import pandas as pd
from mlflow.models import ModelSignature
from mlflow.types import Schema, ColSpec

# Define model signature: input is a string column 'text', output is a dict
input_schema = Schema([ColSpec("string", "text")])
output_schema = Schema([ColSpec("double", "score")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

with mlflow.start_run(run_name="scorer_deploy") as run:
    model = ScorerModel()
    mlflow.pyfunc.log_model(
        "scorer_model",
        python_model=model,
        registered_model_name=MODEL_NAME,
        signature=signature,
    )
    run_id = run.info.run_id
    print(f"Run ID: {{run_id}}")

# Get the latest version
from mlflow.tracking import MlflowClient
client = MlflowClient(registry_uri="databricks-uc")
versions = client.search_model_versions(f"name='{{MODEL_NAME}}'")
latest_version = max(int(v.version) for v in versions)
print(f"Model version: {{latest_version}}")

dbutils.notebook.exit(str(latest_version))
"""

# Upload notebook
notebook_bytes = base64.b64encode(notebook_code.encode()).decode()
w.workspace.import_(
    path=NOTEBOOK_PATH,
    content=notebook_bytes,
    language=Language.PYTHON,
    format=ImportFormat.SOURCE,
    overwrite=True,
)
print(f"Notebook uploaded to {NOTEBOOK_PATH}")

# Submit job with mlflow in environment dependencies
# Using JobEnvironment to pre-install mlflow before notebook runs
print("Submitting job with mlflow dependency...")
job_run = w.jobs.submit(
    run_name=f"{PREFIX}_scorer_deploy_v2",
    environments=[
        jobs.JobEnvironment(
            environment_key="mlflow_env",
            spec=compute.Environment(
                # Use the ML base environment which has MLflow pre-installed
                base_environment="workspace-base-environments/databricks_ml",
            ),
        )
    ],
    tasks=[
        jobs.SubmitTask(
            task_key="deploy",
            environment_key="mlflow_env",
            notebook_task=jobs.NotebookTask(
                notebook_path=NOTEBOOK_PATH,
            ),
        )
    ],
)

run_id = job_run.run_id
print(f"Job run ID: {run_id}")
print("Waiting for job to complete (may take several minutes)...")

result = w.jobs.wait_get_run_job_terminated_or_skipped(
    run_id=run_id,
    timeout=datetime.timedelta(seconds=900),
)
print(f"Job state: {result.state.life_cycle_state}")

if result.state.result_state.value != "SUCCESS":
    for task in (result.tasks or []):
        if task.run_id:
            try:
                output = w.jobs.get_run_output(run_id=task.run_id)
                print(f"Task error: {output.error}")
                if output.error_trace:
                    print(f"Trace: {output.error_trace[:1500]}")
            except Exception as ex:
                print(f"Could not get output: {ex}")
    raise RuntimeError(f"Notebook run failed: {result.state.state_message}")

# Get model version from notebook output
model_version = "1"
try:
    for task in (result.tasks or []):
        if task.run_id and task.notebook_task:
            output = w.jobs.get_run_output(run_id=task.run_id)
            if output.notebook_output and output.notebook_output.result:
                model_version = output.notebook_output.result.strip()
                break
    print(f"Model version: {model_version}")
except Exception as e:
    print(f"Could not get notebook output: {e}")

# =========================================================================
# Create serving endpoint
# =========================================================================

print(f"\nCreating serving endpoint: {ENDPOINT_NAME}")
print(f"Model: {MODEL_NAME} v{model_version}")

existing = None
try:
    existing = w.serving_endpoints.get(ENDPOINT_NAME)
    print(f"Endpoint already exists: {existing.state}")
except Exception:
    existing = None

if existing is None:
    endpoint = w.serving_endpoints.create_and_wait(
        name=ENDPOINT_NAME,
        config=serving.EndpointCoreConfigInput(
            name=ENDPOINT_NAME,
            served_entities=[
                serving.ServedEntityInput(
                    name="scorer",
                    entity_name=MODEL_NAME,
                    entity_version=model_version,
                    workload_size="Small",
                    scale_to_zero_enabled=True,
                )
            ]
        ),
        timeout=datetime.timedelta(seconds=1200),
    )
    print(f"Endpoint created: {endpoint.state}")
else:
    print(f"Using existing endpoint")

# =========================================================================
# Invoke endpoint on payloads
# =========================================================================

print("\nEndpoint ready. Loading payloads...")
with open("data/payloads.json") as f:
    payloads = json.load(f)

print(f"Invoking endpoint on {len(payloads)} payloads...")
responses = []
for i, text in enumerate(payloads):
    result = w.serving_endpoints.query(
        name=ENDPOINT_NAME,
        inputs=[{"text": text}],
    )
    print(f"Payload {i} result: {result}")
    pred = getattr(result, "predictions", None)
    if pred is not None:
        if isinstance(pred, list) and len(pred) > 0:
            item = pred[0]
            if isinstance(item, dict):
                responses.append(item.get("score", item))
            else:
                responses.append(item)
        else:
            responses.append(pred)
    else:
        responses.append(result)

print(f"\nResponses: {responses}")

# Write results
os.makedirs("submission", exist_ok=True)
output = {
    "endpoint_name": ENDPOINT_BASE,
    "responses": responses,
}
with open("submission/answers.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"Written to submission/answers.json")
print(json.dumps(output, indent=2))
