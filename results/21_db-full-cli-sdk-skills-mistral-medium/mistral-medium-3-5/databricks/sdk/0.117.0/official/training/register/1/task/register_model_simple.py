#!/usr/bin/env python3
import os
import json
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ImportFormat

# Read the metrics and model data
with open('data/metrics.json', 'r') as f:
    metrics = json.load(f)

with open('data/model.json', 'r') as f:
    model_data = json.load(f)

print(f"Metrics: {metrics}")

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
        comment=f"Churn prediction model. Metrics: {json.dumps(metrics)}"
    )
    print(f"Created registered model: {model_info.full_name}")
except Exception as e:
    print(f"Registered model may already exist: {e}")
    # Update it with metrics
    try:
        model_info = wc.registered_models.update(
            full_name=f'{catalog_name}.{schema_name}.churnmodel2f9d47',
            comment=f"Churn prediction model. Metrics: {json.dumps(metrics)}"
        )
        print(f"Updated registered model: {model_info.full_name}")
    except Exception as e2:
        print(f"Error updating model: {e2}")

# Upload the model artifact to a volume
prefix = os.environ.get('MLPAB_DATABRICKS_PREFIX', '')
volume_name = f"{prefix}_model_artifacts"

try:
    from databricks.sdk.service.catalog import VolumeType
    volume = wc.volumes.create(
        catalog_name=catalog_name,
        schema_name=schema_name,
        name=volume_name,
        volume_type=VolumeType.MANAGED
    )
    print(f"Created volume: {volume.full_name}")
except Exception as e:
    print(f"Volume may already exist: {e}")

# Upload model artifact to volume
volume_path = f'/Volumes/{catalog_name}/{schema_name}/{volume_name}/model.json'
try:
    with open('data/model.json', 'rb') as f:
        wc.files.upload(volume_path, f)
    print(f"Uploaded model artifact to: {volume_path}")
except Exception as e:
    print(f"Error uploading artifact: {e}")

# Write the answers.json
answers = {
    "model_name": "churnmodel2f9d47",
    "version": 1,
    "metrics": metrics
}

with open('submission/answers.json', 'w') as f:
    json.dump(answers, f)
print(f"Written answers.json: {answers}")
