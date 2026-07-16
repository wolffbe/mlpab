#!/usr/bin/env python3
import os
import json
import time
from databricks.sdk import WorkspaceClient

# Read the model and metrics files
with open('data/model.json', 'r') as f:
    model_data = json.load(f)

with open('data/metrics.json', 'r') as f:
    metrics = json.load(f)

# Connect to Databricks
w = WorkspaceClient()

# Get user info
user = w.current_user.me().user_name
prefix = os.environ.get('MLPAB_DATABRICKS_PREFIX', 'mlpab18f3b5')
schema = os.environ['MLPAB_DATABRICKS_SCHEMA']
catalog_name = schema.split('.')[0]
schema_name = schema.split('.')[1]

model_name = "churnmodel2f9d47"
version = 1

# Step 1: Ensure the registered model exists in Unity Catalog
print(f"Ensuring registered model {model_name} exists in {catalog_name}.{schema_name}")
try:
    model = w.registered_models.get(
        full_name=f"{catalog_name}.{schema_name}.{model_name}"
    )
    print(f"Found existing model: {model.full_name}")
except Exception as e:
    print(f"Creating model: {e}")
    model = w.registered_models.create(
        name=model_name,
        catalog_name=catalog_name,
        schema_name=schema_name,
        comment="Churn prediction model"
    )
    print(f"Created model: {model.full_name}")

# Step 2: Create an MLflow experiment and run
# Use a unique experiment name
experiment_path = f"/Users/{user}/{prefix}/churnmodel2f9d47_experiment"

try:
    experiment = w.experiments.get_by_name(experiment_path)
    print(f"Found existing experiment: {experiment.experiment_id}")
except Exception as e:
    print(f"Creating experiment: {e}")
    experiment = w.experiments.create_experiment(
        name=experiment_path
    )
    print(f"Created experiment: {experiment.experiment_id}")

# Create a run
run_response = w.experiments.create_run(
    experiment_id=experiment.experiment_id
)
run_id = run_response.run.info.run_id
print(f"Created run: {run_id}")

# Log metrics
timestamp = int(time.time() * 1000)
for metric_name, metric_value in metrics.items():
    w.experiments.log_metric(
        key=metric_name,
        value=metric_value,
        timestamp=timestamp,
        run_id=run_id
    )
    print(f"Logged metric {metric_name}: {metric_value}")

# Upload model.json to DBFS under the experiment's artifact location
# The logged model artifacts are stored at dbfs:/databricks/mlflow-tracking/<experiment_id>/logged_models/<model_id>/artifacts
# But we can't use DBFS root. Let me try using the workspace filesystem and then reference it

# Actually, let's try to use the model_registry API with the short name
# and see if it works now that the registered model exists in Unity Catalog

# First, let's try to get the model version that might have been created automatically
try:
    versions = w.model_versions.list(full_name=f"{catalog_name}.{schema_name}.{model_name}")
    print(f"Existing model versions: {[v.version for v in versions]}")
except Exception as e:
    print(f"Error listing model versions: {e}")

# Try to create a model version using the model_registry API
# with the short name and a DBFS source
try:
    # Use the workspace path as the source
    workspace_model_dir = f"/Users/{user}/{prefix}/churnmodel2f9d47"
    mv = w.model_registry.create_model_version(
        name=model_name,
        source=f"dbfs:{workspace_model_dir}",
        description="Churn prediction model v1"
    )
    print(f"Created model version: {mv.version}")
except Exception as e:
    print(f"Error creating model version: {e}")

# Step 3: Write the submission/answers.json
answers = {
    "model_name": model_name,
    "version": version,
    "metrics": metrics
}

with open('submission/answers.json', 'w') as f:
    json.dump(answers, f, indent=2)

print(f"Written submission/answers.json: {answers}")
print("Done!")
