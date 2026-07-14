#!/usr/bin/env python3
"""Script to deploy the scorer as a real-time endpoint and invoke it on payloads."""

import os
import json
import time
from databricks.sdk import WorkspaceClient

# Environment variables
MLPAB_DATABRICKS_SCHEMA = os.environ.get('MLPAB_DATABRICKS_SCHEMA', 'workspace.mlpab222a08')
MLPAB_DATABRICKS_PREFIX = os.environ.get('MLPAB_DATABRICKS_PREFIX', 'mlpab222a08')

# User info
USER_EMAIL = 'benedict@hopsworks.ai'

# Parse schema
catalog_name, schema_name = MLPAB_DATABRICKS_SCHEMA.split('.')

print(f"Catalog: {catalog_name}, Schema: {schema_name}")
print(f"Prefix: {MLPAB_DATABRICKS_PREFIX}")

# Initialize workspace client
wc = WorkspaceClient()

# Step 1: Upload scorer.py to workspace
workspace_dir = f"/Users/{USER_EMAIL}/{MLPAB_DATABRICKS_PREFIX}"
workspace_scorer_path = f"{workspace_dir}/scorer.py"

print(f"Uploading scorer.py to {workspace_scorer_path}")
with open('data/scorer.py', 'rb') as f:
    wc.workspace.upload(workspace_scorer_path, f.read(), overwrite=True)
print("scorer.py uploaded successfully")

# Step 2: Create experiment
print("Creating experiment...")
experiment_name = f"{MLPAB_DATABRICKS_PREFIX}_scorer_experiment"
try:
    experiment = wc.experiments.create_experiment(
        name=experiment_name,
        artifact_location=f"dbfs:/Users/{USER_EMAIL}/{MLPAB_DATABRICKS_PREFIX}/artifacts",
        tags=[{"key": "task", "value": "scorer_deployment"}]
    )
    experiment_id = experiment.experiment_id
    print(f"Experiment created: {experiment_id}")
except Exception as e:
    print(f"Error creating experiment: {e}")
    # Maybe it already exists
    try:
        experiment = wc.experiments.get_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        print(f"Experiment already exists: {experiment_id}")
    except Exception as e2:
        print(f"Error getting experiment: {e2}")
        raise

# Step 3: Create run
print("Creating run...")
run = wc.experiments.create_run(
    experiment_id=experiment_id,
    run_name=f"{MLPAB_DATABRICKS_PREFIX}_scorer_run",
    artifact_uri=f"dbfs:/Users/{USER_EMAIL}/{MLPABRICKS_PREFIX}/artifacts",
    tags=[{"key": "task", "value": "scorer_deployment"}]
)
run_id = run.run_id
print(f"Run created: {run_id}")

# Step 4: Upload model files to the run's artifact location
# The artifact location is typically dbfs:/databricks-datasets/mlflow/...
# But we can use our own location

# Upload scorer.py to artifact location
artifact_scorer_path = f"dbfs:/Users/{USER_EMAIL}/{MLPAB_DATABRICKS_PREFIX}/artifacts/{run_id}/artifacts/scorer.py"
print(f"Uploading scorer.py to artifact location: {artifact_scorer_path}")
with open('data/scorer.py', 'rb') as f:
    wc.dbfs.upload(artifact_scorer_path, f.read(), overwrite=True)
print("scorer.py uploaded to artifact location")

# Step 5: Create MLmodel JSON
mlmodel_json = {
    "flavors": {
        "python_function": {
            "loader_module": "mlflow.pyfunc",
            "python_version": "3.9",
            "code": "scorer.py"
        }
    },
    "signature": {
        "inputs": "[{'type': 'string', 'name': 'text'}]",
        "outputs": "[{'type': 'double', 'name': 'score'}]"
    },
    "model_uuid": "local-model",
    "run_id": run_id
}

# Step 6: Log the model using the experiments API
print("Logging model...")
try:
    # Use the log_model method with the model JSON
    model_response = wc.experiments.log_model(
        model_json=json.dumps(mlmodel_json),
        run_id=run_id
    )
    print(f"Model logged: {model_response}")
except Exception as e:
    print(f"Error logging model: {e}")
    raise

# Step 7: Create registered model in Unity Catalog
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
    print(f"Error creating registered model: {e}")
    # Maybe it already exists
    try:
        registered_model = wc.registered_models.get(full_model_name)
        print(f"Registered model already exists: {registered_model}")
    except Exception as e2:
        print(f"Error getting registered model: {e2}")
        raise

# Step 8: Create model version
print("Creating model version...")
# The source should be the run URI
model_source = f"runs:/{run_id}/model"

try:
    model_version = wc.model_registry.create_model_version(
        name=full_model_name,
        source=model_source,
        description="Initial version of scorer model"
    )
    print(f"Model version created: {model_version}")
except Exception as e:
    print(f"Error creating model version: {e}")
    raise

# Step 9: Deploy as serving endpoint
endpoint_name = "scorer40bb09"

print(f"Creating serving endpoint: {endpoint_name}")

from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedModelInput
)

config = EndpointCoreConfigInput(
    name=endpoint_name,
    served_models=[
        ServedModelInput(
            model_name=full_model_name,
            model_version="1",
            workload_size="SMALL",
            scale_to_zero_enabled=True,
            min_provisioned_concurrency=0
        )
    ]
)

try:
    endpoint = wc.serving_endpoints.create_and_wait(
        name=endpoint_name,
        config=config,
        timeout=1200
    )
    print(f"Serving endpoint created: {endpoint}")
except Exception as e:
    print(f"Error creating serving endpoint: {e}")
    raise

# Step 10: Load payloads
with open('data/payloads.json', 'r') as f:
    payloads = json.load(f)

print(f"Loaded {len(payloads)} payloads")

# Step 11: Invoke endpoint on each payload
responses = []
for i, payload in enumerate(payloads):
    print(f"Invoking endpoint for payload {i+1}/{len(payloads)}")
    try:
        response = wc.serving_endpoints.query(
            name=endpoint_name,
            inputs={"text": payload}
        )
        print(f"Response: {response}")
        responses.append(response)
    except Exception as e:
        print(f"Error invoking endpoint: {e}")
        raise

# Step 12: Write results to submission/answers.json
results = {
    "endpoint_name": endpoint_name,
    "responses": responses
}

os.makedirs('submission', exist_ok=True)
with open('submission/answers.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Results written to submission/answers.json")
print("Done!")
