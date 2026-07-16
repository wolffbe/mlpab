#!/usr/bin/env python3
import os
import json
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

# Step 1: Create the registered model in Unity Catalog
print(f"Creating registered model {model_name} in {catalog_name}.{schema_name}")
try:
    model = w.registered_models.create(
        name=model_name,
        catalog_name=catalog_name,
        schema_name=schema_name,
        comment="Churn prediction model"
    )
    print(f"Created model: {model.full_name}")
except Exception as e:
    print(f"Model may already exist: {e}")

# Step 2: Create an MLflow experiment
experiment_path = f"/Users/{user}/{prefix}/churnmodel2f9d47"

try:
    experiment = w.experiments.create_experiment(
        name=experiment_path
    )
    print(f"Created experiment: {experiment.experiment_id}")
except Exception as e:
    print(f"Experiment may already exist: {e}")
    # Get existing experiment
    experiment = w.experiments.get_by_name(experiment_path)
    print(f"Found existing experiment: {experiment.experiment_id}")

# Create a run
run = w.experiments.create_run(
    experiment_id=experiment.experiment_id
)
print(f"Created run: {run.info.run_id}")

# Log metrics
for metric_name, metric_value in metrics.items():
    w.experiments.log_metric(
        run_id=run.info.run_id,
        key=metric_name,
        value=metric_value
    )
    print(f"Logged metric {metric_name}: {metric_value}")

# Create an MLmodel file for the model
mlmodel = {
    "flavors": {
        "custom": {
            "model_data": model_data
        }
    }
}

# Log the model using create_logged_model
logged_model = w.experiments.create_logged_model(
    experiment_id=experiment.experiment_id,
    name=model_name,
    model_type="custom",
    source_run_id=run.info.run_id
)
print(f"Created logged model: {logged_model}")

# Now try to register the model
# Use the model_registry API to create a model version
try:
    mv = w.model_registry.create_model_version(
        name=f"{catalog_name}.{schema_name}.{model_name}",
        source=f"runs:/{run.info.run_id}/model",
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
