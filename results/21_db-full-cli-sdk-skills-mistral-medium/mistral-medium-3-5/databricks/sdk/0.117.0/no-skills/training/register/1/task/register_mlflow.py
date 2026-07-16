#!/usr/bin/env python3
import os
import json
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ImportFormat

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
schema = os.environ['MLPABRICKS_SCHEMA']
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
    # Try to get existing model
    try:
        model = w.registered_models.get(
            full_name=f"{catalog_name}.{schema_name}.{model_name}"
        )
        print(f"Found existing model: {model.full_name}")
    except Exception as e2:
        print(f"Failed to get model: {e2}")

# Step 2: Create an MLflow experiment and run
# Set up the experiment path
experiment_path = f"/Users/{user}/{prefix}/churnmodel2f9d47"

# Create or get the experiment
experiment = w.experiments.create_experiment(
    name=experiment_path
)
print(f"Created experiment: {experiment.experiment_id}")

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
        value=metric_value,
        timestamp=None,
        step=None
    )
    print(f"Logged metric {metric_name}: {metric_value}")

# Log the model
# The log_model method expects a model URI or creates a model artifact
# Let's try to log the model data as a custom model
w.experiments.log_model(
    run_id=run.info.run_id,
    model_name=model_name,
    model_json=model_data,
    flavor=None,
    registered_model_name=f"{catalog_name}.{schema_name}.{model_name}"
)
print(f"Logged model")

# Finalize the run
w.experiments.finalize_logged_model(
    run_id=run.info.run_id,
    registered_model_name=f"{catalog_name}.{schema_name}.{model_name}"
)
print(f"Finalized logged model")

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
