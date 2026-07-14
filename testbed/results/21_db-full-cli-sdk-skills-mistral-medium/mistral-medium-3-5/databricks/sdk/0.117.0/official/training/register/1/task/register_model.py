#!/usr/bin/env python3
import os
import json
from databricks.sdk import WorkspaceClient

# Read the metrics and model data
with open('data/metrics.json', 'r') as f:
    metrics = json.load(f)

with open('data/model.json', 'r') as f:
    model_data = json.load(f)

print(f"Metrics: {metrics}")
print(f"Model data: {model_data}")

# Check the schema
schema = os.environ.get('MLPAB_DATABRICKS_SCHEMA', '')
print(f"Schema: {schema}")

# Split into catalog and schema
parts = schema.split('.')
catalog_name = parts[0]
schema_name = parts[1]
print(f"Catalog: {catalog_name}, Schema: {schema_name}")

# Initialize workspace client
wc = WorkspaceClient()

# Create a registered model in Unity Catalog
try:
    model_info = wc.registered_models.create(
        catalog_name=catalog_name,
        schema_name=schema_name,
        name="churnmodel2f9d47",
        comment="Churn prediction model"
    )
    print(f"Created registered model: {model_info}")
except Exception as e:
    print(f"Registered model may already exist: {e}")
    # Continue anyway, we'll create the version

# Now upload the model artifact and create a model version
# First, upload the model.json file to workspace
from databricks.sdk.service.workspace import ImportFormat
prefix = os.environ.get('MLPAB_DATABRICKS_PREFIX', '')
user = wc.current_user.me().user_name
# Use workspace path format
workspace_dir = f"/Users/{user}/{prefix}"
workspace_path = f"{workspace_dir}/model.json"

print(f"Uploading model to: {workspace_path}")
# Create directory if it doesn't exist
try:
    wc.workspace.mkdirs(workspace_dir)
    print(f"Created directory: {workspace_dir}")
except Exception as e:
    print(f"Directory may already exist: {e}")

# Read the file content
with open('data/model.json', 'rb') as f:
    model_content = f.read()

# Upload to workspace using RAW format
wc.workspace.upload(workspace_path, model_content, format=ImportFormat.RAW, overwrite=True)
print(f"Uploaded model to workspace")

# Create a model version
try:
    # Use the workspace path as the source
    # The model registry expects a URI, so we need to convert workspace path to a URI
    # For workspace files, we can use the workspace:// URI scheme
    source_path = f"workspace{workspace_path}"
    mv_response = wc.model_registry.create_model_version(
        name="churnmodel2f9d47",
        source=source_path,
        description="Version 1 of churn prediction model",
        tags=[
            {"key": "auc", "value": str(metrics.get("auc", ""))},
            {"key": "accuracy", "value": str(metrics.get("accuracy", ""))}
        ]
    )
    print(f"Created model version: {mv_response}")
    
    # Write the answers.json
    answers = {
        "model_name": "churnmodel2f9d47",
        "version": 1,
        "metrics": metrics
    }
    
    with open('submission/answers.json', 'w') as f:
        json.dump(answers, f)
    print(f"Written answers.json: {answers}")
    
except Exception as e:
    print(f"Error creating model version: {e}")
