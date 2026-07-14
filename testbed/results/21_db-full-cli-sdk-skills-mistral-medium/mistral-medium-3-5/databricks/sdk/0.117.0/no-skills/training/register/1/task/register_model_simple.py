#!/usr/bin/env python3
import os
import json
from databricks.sdk import WorkspaceClient

# Read the metrics file
with open('data/metrics.json', 'r') as f:
    metrics = json.load(f)

# Connect to Databricks
w = WorkspaceClient()

# Schema from environment
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
    print(f"Error creating model: {e}")
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
        raise

# Step 2: Upload the model artifact to DBFS
# We need to upload data/model.json to a location that can be referenced as the source
import tempfile
import base64

# Read the model file
with open('data/model.json', 'r') as f:
    model_data = f.read()

# Upload to DBFS
# First, let's create a path in DBFS for the model
prefix = os.environ.get('MLPAB_DATABRICKS_PREFIX', 'mlpab')
dbfs_path = f"dbfs:/Users/{w.current_user.me().user_name}/{prefix}/churnmodel2f9d47/model.json"

# Write to DBFS
w.dbfs.put_file(dbfs_path, model_data.encode('utf-8'), overwrite=True)
print(f"Uploaded model to: {dbfs_path}")

# Step 3: Create a model version
# The source should be the DBFS path
# For Unity Catalog, we need to use the model_versions API or model_registry API
# Let's try using the model_registry API with the full name

full_model_name = f"{catalog_name}.{schema_name}.{model_name}"
print(f"Creating model version for: {full_model_name}")

try:
    mv = w.model_registry.create_model_version(
        name=full_model_name,
        source=dbfs_path,
        description="Churn prediction model v1"
    )
    print(f"Created model version: {mv.version}")
except Exception as e:
    print(f"Error creating model version with model_registry: {e}")
    # Try with just the model name
    try:
        mv = w.model_registry.create_model_version(
            name=model_name,
            source=dbfs_path,
            description="Churn prediction model v1"
        )
        print(f"Created model version: {mv.version}")
    except Exception as e2:
        print(f"Error creating model version with short name: {e2}")
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
