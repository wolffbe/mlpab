#!/usr/bin/env python3
"""Script to register the fine-tuned model in Unity Catalog using MLflow."""

import os
import json
import mlflow
from mlflow.tracking import MlflowClient

def main():
    # Set up MLflow to use Unity Catalog
    mlflow.set_registry_uri("databricks-uc")
    
    # Load metrics from the job output
    with open("/dbfs/Users/benedict@logicalclocks.com/mlpabb86f4f/output/metrics.json", "r") as f:
        metrics = json.load(f)
    
    # Register the model in Unity Catalog
    model_name = "workspace.mlpabb86f4f.ftmodel65e929"
    model_uri = "dbfs:/Users/benedict@logicalclocks.com/mlpabb86f4f/output/finetuned_model.npz"
    
    # Log the model with metrics
    with mlflow.start_run() as run:
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log the model
        mlflow.log_artifact("/dbfs/Users/benedict@logicalclocks.com/mlpabb86f4f/output/finetuned_model.npz")
        
        # Register the model
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
            tags={"version": "1"}
        )
        
        print(f"Registered model version: {model_version}")

if __name__ == "__main__":
    main()