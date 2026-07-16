#!/usr/bin/env python3
import os
import json
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ImportFormat

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
            full_name=f"{catalog_name}.{schema_name}.{model_name}"
        )
        print(f"Found existing model: {model.full_name}")
    except Exception as e2:
        print(f"Failed to get model: {e2}")

# Step 2: Upload the model artifact to workspace filesystem
workspace_model_dir = f"/Users/{user}/{prefix}/churnmodel2f9d47"
w.workspace.mkdirs(workspace_model_dir)

# Upload model.json
model_path = f"{workspace_model_dir}/model.json"
with open('data/model.json', 'rb') as f:
    w.workspace.upload(model_path, f, format=ImportFormat.RAW, overwrite=True)
print(f"Uploaded model to: {model_path}")

# Upload metrics.json
metrics_path = f"{workspace_model_dir}/metrics.json"
with open('data/metrics.json', 'rb') as f:
    w.workspace.upload(metrics_path, f, format=ImportFormat.RAW, overwrite=True)
print(f"Uploaded metrics to: {metrics_path}")

# Step 3: Create a model version using the Unity Catalog API
full_model_name = f"{catalog_name}.{schema_name}.{model_name}"

# Use the raw API client
api_client = w._api

# Create model version request body
body = {
    "version": version,
    "source": f"dbfs:{workspace_model_dir}",
    "description": "Churn prediction model v1"
}

# Try to create the model version
try:
    response = api_client.do(
        "POST",
        f"/api/2.1/unity-catalog/models/{full_model_name}/versions",
        body=body
    )
    print(f"Created model version: {response}")
except Exception as e:
    print(f"Error creating model version with UC API: {e}")
    # Try without the dbfs: prefix
    try:
        body["source"] = workspace_model_dir
        response = api_client.do(
            "POST",
            f"/api/2.1/unity-catalog/models/{full_model_name}/versions",
            body=body
        )
        print(f"Created model version: {response}")
    except Exception as e2:
        print(f"Error without dbfs prefix: {e2}")
        # Try with file: prefix
        try:
            body["source"] = f"file:{workspace_model_dir}"
            response = api_client.do(
                "POST",
                f"/api/2.1/unity-catalog/models/{full_model_name}/versions",
                body=body
            )
            print(f"Created model version: {response}")
        except Exception as e3:
            print(f"Error with file prefix: {e3}")
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
