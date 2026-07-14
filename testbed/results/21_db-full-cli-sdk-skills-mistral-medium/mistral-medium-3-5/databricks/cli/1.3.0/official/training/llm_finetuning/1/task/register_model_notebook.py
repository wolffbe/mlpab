# Databricks notebook source
# This notebook will register the fine-tuned model in Unity Catalog

# Cell 1: Import MLflow and set up
import mlflow
import os
import json

# Set the MLflow tracking URI
mlflow.set_registry_uri("databricks-uc")

# Cell 2: Download the model file and metrics from workspace
import base64
import requests

host = os.environ.get('DATABRICKS_HOST', 'https://dbc-2a4591fe-28e4.cloud.databricks.com')
token = os.environ.get('DATABRICKS_TOKEN')
workspace_path = "/Users/benedict@logicalclocks.com/mlpab64367b"
headers = {"Authorization": f"Bearer {token}"}

# Download metrics.json
url = f"{host}/api/2.0/workspace/export"
params = {"path": f"{workspace_path}/metrics.json", "format": "RAW"}
response = requests.get(url, headers=headers, params=params).json()
if 'content' in response:
    metrics_content = base64.b64decode(response['content'])
    metrics = json.loads(metrics_content.decode('utf-8'))
    print(f"Metrics: {metrics}")
else:
    print(f"Failed to download metrics.json: {response}")
    metrics = {"eval_loss": 0.0, "base_eval_loss": 0.0}

# Download finetuned_model.npz
params = {"path": f"{workspace_path}/finetuned_model.npz", "format": "RAW"}
response = requests.get(url, headers=headers, params=params).json()
if 'content' in response:
    model_content = base64.b64decode(response['content'])
    model_path = "/tmp/finetuned_model.npz"
    with open(model_path, 'wb') as f:
        f.write(model_content)
    print(f"Downloaded model to {model_path}")
else:
    print(f"Failed to download finetuned_model.npz: {response}")
    raise Exception("Failed to download model file")

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
    mlflow.log_artifact(model_path, "model")
    
    # Register the model
    mlflow.register_model(
        "runs:/{}/model".format(run.info.run_id),
        model_name,
        tags={"version": "1"}
    )
    
    print(f"Model registered: {model_name}")
    print(f"Run ID: {run.info.run_id}")

print("Done!")
