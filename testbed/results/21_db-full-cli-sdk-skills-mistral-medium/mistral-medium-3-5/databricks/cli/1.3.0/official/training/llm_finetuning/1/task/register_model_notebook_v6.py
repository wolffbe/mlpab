# Databricks notebook source
# This notebook will register the fine-tuned model in Unity Catalog

# Cell 1: Import MLflow and set up
import mlflow

# Set the MLflow tracking URI
mlflow.set_registry_uri("databricks-uc")

# Cell 2: Download files from workspace to /tmp
import os
import json
import base64
import requests

os.makedirs('/tmp', exist_ok=True)

host = os.environ.get('DATABRICKS_HOST', 'https://dbc-2a4591fe-28e4.cloud.databricks.com')
token = os.environ.get('DATABRICKS_TOKEN', '')
workspace_path = "/Users/benedict@logicalclocks.com/mlpab64367b"

# Try without token first (implicit auth in notebook)
headers = {"Authorization": f"Bearer {token}"} if token else {}

# Download metrics.json
url = f"{host}/api/2.0/workspace/export"
params = {"path": f"{workspace_path}/metrics.json", "format": "RAW"}
response = requests.get(url, headers=headers, params=params)
print(f"Response status: {response.status_code}")
print(f"Response text: {response.text[:200]}")

if response.status_code == 200:
    data = response.json()
    if 'content' in data:
        metrics_content = base64.b64decode(data['content'])
        metrics = json.loads(metrics_content.decode('utf-8'))
        print(f"Metrics: {metrics}")
        # Save to /tmp
        with open('/tmp/metrics.json', 'w') as f:
            json.dump(metrics, f)
    else:
        print(f"No content in response: {data}")
        raise Exception("Failed to download metrics")
else:
    print(f"HTTP {response.status_code}: {response.text}")
    raise Exception("Failed to download metrics")

# Download finetuned_model.npz
params = {"path": f"{workspace_path}/finetuned_model.npz", "format": "RAW"}
response = requests.get(url, headers=headers, params=params)
print(f"Model response status: {response.status_code}")

if response.status_code == 200:
    data = response.json()
    if 'content' in data:
        model_content = base64.b64decode(data['content'])
        with open('/tmp/finetuned_model.npz', 'wb') as f:
            f.write(model_content)
        print(f"Downloaded model to /tmp/finetuned_model.npz")
    else:
        print(f"No content in response: {data}")
        raise Exception("Failed to download model")
else:
    print(f"HTTP {response.status_code}: {response.text}")
    raise Exception("Failed to download model")

# Cell 3: Register the model in Unity Catalog
model_name = "workspace.mlpab64367b.ftmodel65e929"

# Start an MLflow run
with mlflow.start_run() as run:
    # Log metrics
    mlflow.log_metrics({
        "eval_loss": metrics["eval_loss"],
        "base_eval_loss": metrics["base_eval_loss"]
    })
    
    # Log the model file
    mlflow.log_artifact("/tmp/finetuned_model.npz", "model")
    
    # Register the model
    mlflow.register_model(
        "runs:/{}/model".format(run.info.run_id),
        model_name,
        tags={"version": "1"}
    )
    
    print(f"Model registered: {model_name}")
    print(f"Run ID: {run.info.run_id}")

print("Done!")
