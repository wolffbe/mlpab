"""
Deploy scorer.py as a custom MLflow pyfunc model on Databricks,
register it in Unity Catalog, create a serving endpoint, and
query it with the payloads from data/payloads.json.
"""
import base64
import datetime
import json
import os
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs as jobs_service
from databricks.sdk.service.compute import ClusterSpec
from databricks.sdk.service.workspace import Language
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
    TrafficConfig,
    Route,
)

w = WorkspaceClient()

SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]   # workspace.mlpab996116
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]   # mlpab996116
USER = w.current_user.me().user_name             # benedict@logicalclocks.com

CATALOG, SCHEMA_NAME = SCHEMA.split(".")         # workspace, mlpab996116
MODEL_NAME = f"{CATALOG}.{SCHEMA_NAME}.scorer98c848"
ENDPOINT_NAME = f"{PREFIX}_scorer98c848"
EXPERIMENT_PATH = f"/Users/{USER}/{PREFIX}/scorer98c848_experiment"
NOTEBOOK_PATH = f"/Users/{USER}/{PREFIX}/scorer98c848_register"
SERVED_ENTITY_NAME = "scorer-entity"

print(f"Schema:   {SCHEMA}")
print(f"Model:    {MODEL_NAME}")
print(f"Endpoint: {ENDPOINT_NAME}")
print(f"User:     {USER}")

# --- Step 1: Create workspace directory and upload scorer.py ---
with open("data/scorer.py", "r") as f:
    scorer_content = f.read()

print("\n[1] Setting up workspace directory...")
w.workspace.mkdirs(f"/Users/{USER}/{PREFIX}")

# --- Step 2: Create the registration notebook ---
# This notebook runs on the cluster and logs/registers the MLflow model.
notebook_src = f'''# Databricks notebook source
# Register scorer as MLflow pyfunc model in Unity Catalog
# COMMAND ----------
import mlflow
import mlflow.pyfunc
import pandas as pd
import math

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment("{EXPERIMENT_PATH}")

# COMMAND ----------
class ScorerModel(mlflow.pyfunc.PythonModel):
    """Wraps the deterministic trigram scorer."""

    def predict(self, context, model_input, params=None):
        import math
        A, B, C, D = 2.158311, 2.825397, 1.859128, -0.317766

        def _w(tri):
            o0, o1, o2 = (ord(ch) for ch in tri)
            return math.sin(A * o0 + B * o1 + C * o2 + D)

        def score(text):
            ll = 0.0
            for i in range(len(text) - 2):
                ll += _w(text[i:i+3])
            return {{"score": round(ll, 6)}}

        if isinstance(model_input, pd.DataFrame):
            results = []
            for _, row in model_input.iterrows():
                text = str(row.iloc[0])
                results.append(score(text))
            return pd.DataFrame(results)
        elif isinstance(model_input, list):
            return [score(str(t)) for t in model_input]
        else:
            return score(str(model_input))

# COMMAND ----------
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

input_schema  = Schema([ColSpec("string", "text")])
output_schema = Schema([ColSpec("double", "score")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

sample_input = pd.DataFrame({{"text": ["hello world"]}})

with mlflow.start_run() as run:
    model_info = mlflow.pyfunc.log_model(
        artifact_path="scorer_model",
        python_model=ScorerModel(),
        signature=signature,
        input_example=sample_input,
        pip_requirements=[],
        registered_model_name="{MODEL_NAME}",
    )
    print(f"Logged model: {{model_info.model_uri}}")

print("Registration complete.")

# COMMAND ----------
from mlflow.tracking import MlflowClient
client = MlflowClient(registry_uri="databricks-uc")
versions = client.search_model_versions(f"name=\\'{MODEL_NAME}\\'")
latest = max([int(v.version) for v in versions])
print(f"Latest version: {{latest}}")
dbutils.notebook.exit(str(latest))
'''

print("[2] Uploading registration notebook...")
w.workspace.import_(
    path=NOTEBOOK_PATH,
    language=Language.PYTHON,
    content=base64.b64encode(notebook_src.encode()).decode(),
    overwrite=True,
)
print(f"    Notebook: {NOTEBOOK_PATH}")

# --- Step 3: Run notebook as a one-time job run ---
print("\n[3] Submitting registration job run (serverless)...")

ENV_KEY = "default"
from databricks.sdk.service.jobs import JobEnvironment
from databricks.sdk.service.compute import Environment

run_result = w.jobs.submit(
    run_name=f"{PREFIX}_scorer98c848_register",
    environments=[
        JobEnvironment(
            environment_key=ENV_KEY,
            spec=Environment(
                client="1",
                dependencies=["mlflow"],
            ),
        )
    ],
    tasks=[
        jobs_service.SubmitTask(
            task_key="register",
            environment_key=ENV_KEY,
            notebook_task=jobs_service.NotebookTask(
                notebook_path=NOTEBOOK_PATH,
                source=jobs_service.Source.WORKSPACE,
            ),
        )
    ],
)

run_id = run_result.run_id
print(f"    Run ID: {run_id}")

# --- Step 4: Wait for job to complete ---
print("\n[4] Waiting for registration job...")
start = time.time()
while True:
    run_state = w.jobs.get_run(run_id=run_id)
    lc = run_state.state.life_cycle_state.value
    elapsed = int(time.time() - start)
    print(f"    [{elapsed}s] {lc}")
    if lc in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        rs = run_state.state.result_state
        if rs and rs.value != "SUCCESS":
            # Print error output for debugging
            if run_state.tasks:
                out = w.jobs.get_run_output(run_id=run_state.tasks[0].run_id)
                if out.error:
                    print(f"    ERROR: {out.error}")
                if out.notebook_output:
                    print(f"    OUTPUT: {out.notebook_output.result}")
            raise RuntimeError(f"Registration job failed: {rs.value}")
        break
    time.sleep(20)

model_version = "1"
if run_state.tasks:
    out = w.jobs.get_run_output(run_id=run_state.tasks[0].run_id)
    if out.notebook_output and out.notebook_output.result:
        model_version = out.notebook_output.result.strip()
        print(f"    Model version: {model_version}")

print(f"\n    Registered: {MODEL_NAME} v{model_version}")

# --- Step 5: Create serving endpoint ---
print(f"\n[5] Creating serving endpoint: {ENDPOINT_NAME}")

# Delete if exists
try:
    w.serving_endpoints.get(name=ENDPOINT_NAME)
    print("    Deleting existing endpoint...")
    w.serving_endpoints.delete(name=ENDPOINT_NAME)
    time.sleep(10)
except Exception:
    pass

endpoint = w.serving_endpoints.create_and_wait(
    name=ENDPOINT_NAME,
    config=EndpointCoreConfigInput(
        name=ENDPOINT_NAME,
        served_entities=[
            ServedEntityInput(
                name=SERVED_ENTITY_NAME,
                entity_name=MODEL_NAME,
                entity_version=model_version,
                workload_size="Small",
                scale_to_zero_enabled=True,
            )
        ],
        traffic_config=TrafficConfig(
            routes=[
                Route(
                    served_entity_name=SERVED_ENTITY_NAME,
                    traffic_percentage=100,
                )
            ]
        ),
    ),
    timeout=datetime.timedelta(minutes=30),
)
print(f"    State: {endpoint.state}")

# --- Step 6: Query the endpoint with each payload ---
with open("data/payloads.json") as f:
    payloads = json.load(f)

print(f"\n[6] Querying endpoint with {len(payloads)} payloads...")
responses = []

for i, text in enumerate(payloads):
    result = w.serving_endpoints.query(
        name=ENDPOINT_NAME,
        dataframe_records=[{"text": text}],
    )
    # Extract the score from the response
    preds = result.predictions
    if isinstance(preds, list) and len(preds) > 0:
        p = preds[0]
        if isinstance(p, dict):
            score_val = p.get("score", p)
        else:
            score_val = p
    else:
        score_val = preds

    print(f"    [{i+1}] score={score_val}")
    responses.append(score_val)

# --- Step 7: Write answers.json ---
answers = {
    "endpoint_name": "scorer98c848",
    "responses": responses,
}

os.makedirs("submission", exist_ok=True)
with open("submission/answers.json", "w") as f:
    json.dump(answers, f, indent=2)

print(f"\n[7] Written submission/answers.json")
print(json.dumps(answers, indent=2))
