#!/usr/bin/env python3
import os
import json
import base64
from databricks.sdk import WorkspaceClient

# Read the model and metrics files
with open('data/model.json', 'r') as f:
    model_data = f.read()

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
    # Try to get existing model
    try:
        model = w.registered_models.get(
            catalog_name=catalog_name,
            schema_name=schema_name,
            name=model_name
        )
        print(f"Found existing model: {model.full_name}")
    except Exception as e2:
        print(f"Failed to get model: {e2}")

# Step 2: Upload the model artifact to workspace filesystem
# Create a path for the model artifacts
workspace_model_dir = f"/Users/{user}/{prefix}/churnmodel2f9d47"

# Ensure directory exists
w.workspace.mkdirs(workspace_model_dir)

# Upload model.json
model_path = f"{workspace_model_dir}/model.json"
w.workspace.upload(model_path, "data/model.json", overwrite=True)
print(f"Uploaded model to: {model_path}")

# Upload metrics.json
metrics_path = f"{workspace_model_dir}/metrics.json"
w.workspace.upload(metrics_path, "data/metrics.json", overwrite=True)
print(f"Uploaded metrics to: {metrics_path}")

# Step 3: Create a model version
# The source should point to the workspace directory containing the model
# For workspace files, the URI might be different
# Let's try using the workspace path directly
source_uri = f"{workspace_model_dir}"

print(f"Creating model version with source: {source_uri}")
try:
    mv = w.model_registry.create_model_version(
        name=f"{catalog_name}.{schema_name}.{model_name}",
        source=source_uri,
        description="Churn prediction model v1"
    )
    print(f"Created model version: {mv.version}")
except Exception as e:
    print(f"Error with full name: {e}")
    # Try with just the model name
    try:
        mv = w.model_registry.create_model_version(
            name=model_name,
            source=source_uri,
            description="Churn prediction model v1"
        )
        print(f"Created model version: {mv.version}")
    except Exception as e2:
        print(f"Error with short name: {e2}")
        # Try with dbfs: prefix
        try:
            mv = w.model_registry.create_model_version(
                name=model_name,
                source=f"dbfs:{workspace_model_dir}",
                description="Churn prediction model v1"
            )
            print(f"Created model version: {mv.version}")
        except Exception as e3:
            print(f"Error with dbfs prefix: {e3}")
            raise

# Step 4: Write the submission/answers.json
answers = {
    "model_name": model_name,
    "version": version,
    "metrics": metrics
}

with open('submission/answers.json', 'w') as f:
    json.dump(answers, f, indent=2)

print(f"Written submission/answers.json: {answers}")
print("Done!")
