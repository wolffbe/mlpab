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

# Schema from environment
schema = os.environ['MLPAB_DATABRICKS_SCHEMA']

# Register the model
model_name = "churnmodel2f9d47"
version = 1

# Create the model in the registry
# The catalog.schema format for the model
catalog_schema = schema

# First, ensure the catalog exists, then create the model
try:
    # Create or get the model
    model = w.model_registry.create_model(
        name=model_name,
        catalog_name=catalog_schema.split('.')[0],  # "workspace"
        schema_name=catalog_schema.split('.')[1],   # "mlpab18f3b5"
        description="Churn prediction model"
    )
    print(f"Created model: {model.full_name}")
except Exception as e:
    print(f"Model may already exist: {e}")
    # Try to get existing model
    try:
        model = w.model_registry.get_model(
            catalog_name=catalog_schema.split('.')[0],
            schema_name=catalog_schema.split('.')[1],
            name=model_name
        )
        print(f"Found existing model: {model.full_name}")
    except Exception as e2:
        print(f"Failed to get model: {e2}")
        raise

# Upload the model artifact
# We need to upload the model.json file to MLflow first, then register it
# Let's use MLflow to log the model and metrics

from databricks.sdk.service import ml

# Create a model version with the artifact
# First, let's upload the model file to dbfs or use MLflow
import mlflow

# Set the MLflow tracking URI to Databricks
mlflow.set_experiment(f"/Users/{w.current_user.me().user_name}/mlpab18f3b5/churnmodel2f9d47")

# Start a run
with mlflow.start_run() as run:
    # Log the metrics
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)
    
    # Log the model artifact
    mlflow.log_dict(model_data, "model.json")
    
    # Get the run ID
    run_id = run.info.run_id
    print(f"MLflow run ID: {run_id}")
    
    # Now register the model in the model registry
    # Use the MLflow run to create a model version
    model_uri = f"runs:/{run_id}/model"
    
    # Register the model version
    mv = w.model_registry.create_model_version(
        name=model_name,
        catalog_name=catalog_schema.split('.')[0],
        schema_name=catalog_schema.split('.')[1],
        source=model_uri,
        version=version
    )
    print(f"Created model version: {mv.version}")

# Now write the submission/answers.json
answers = {
    "model_name": model_name,
    "version": version,
    "metrics": metrics
}

with open('submission/answers.json', 'w') as f:
    json.dump(answers, f, indent=2)

print(f"Written submission/answers.json: {answers}")
print("Done!")
