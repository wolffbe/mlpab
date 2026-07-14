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

# Step 2: Upload the model artifact to DBFS
# Create a path for the model artifacts
dbfs_model_dir = f"/Users/{user}/{prefix}/churnmodel2f9d47"
dbfs_model_path = f"{dbfs_model_dir}/model.json"

# Ensure directory exists
w.dbfs.mkdirs(dbfs_model_dir)

# Upload model.json
# The put method expects contents as base64 string
model_data_b64 = base64.b64encode(model_data.encode('utf-8')).decode('utf-8')
w.dbfs.put(path=dbfs_model_path, contents=model_data_b64, overwrite=True)
print(f"Uploaded model to: {dbfs_model_path}")

# Step 3: Upload metrics.json as well (for reference)
with open('data/metrics.json', 'r') as f:
    metrics_data = f.read()
metrics_data_b64 = base64.b64encode(metrics_data.encode('utf-8')).decode('utf-8')
w.dbfs.put(path=f"{dbfs_model_dir}/metrics.json", contents=metrics_data_b64, overwrite=True)
print(f"Uploaded metrics to: {dbfs_model_dir}/metrics.json")

# Step 4: Create a model version
# The source should point to the DBFS directory containing the model
source_uri = f"dbfs:/{dbfs_model_dir}"

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
        raise

# Step 5: Write the submission/answers.json
answers = {
    "model_name": model_name,
    "version": version,
    "metrics": metrics
}

with open('submission/answers.json', 'w') as f:
    json.dump(answers, f, indent=2)

print(f"Written submission/answers.json: {answers}")
print("Done!")
