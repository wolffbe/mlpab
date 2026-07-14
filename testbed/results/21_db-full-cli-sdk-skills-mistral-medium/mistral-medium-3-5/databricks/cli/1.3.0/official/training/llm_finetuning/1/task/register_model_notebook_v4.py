# Databricks notebook source
# This notebook will register the fine-tuned model in Unity Catalog

# Cell 1: Import MLflow and set up
import mlflow

# Set the MLflow tracking URI
mlflow.set_registry_uri("databricks-uc")

# Cell 2: Read the model file and metrics from workspace
# The files are already in the workspace
import json

# Read metrics.json from workspace
with open("/Workspace/Users/benedict@logicalclocks.com/mlpab64367b/metrics.json", "r") as f:
    metrics = json.load(f)
    print(f"Metrics: {metrics}")

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
    mlflow.log_artifact("/Workspace/Users/benedict@logicalclocks.com/mlpab64367b/finetuned_model.npz", "model")
    
    # Register the model
    mlflow.register_model(
        "runs:/{}/model".format(run.info.run_id),
        model_name,
        tags={"version": "1"}
    )
    
    print(f"Model registered: {model_name}")
    print(f"Run ID: {run.info.run_id}")

print("Done!")
