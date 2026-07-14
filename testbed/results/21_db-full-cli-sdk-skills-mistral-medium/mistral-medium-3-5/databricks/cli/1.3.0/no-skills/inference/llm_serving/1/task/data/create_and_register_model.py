# Databricks notebook source
# MAGIC %md
# MAGIC Create and register model for scorer40bb09

# COMMAND ----------

import mlflow
import sys
sys.path.insert(0, "/dbfs/Volumes/workspace/mlpabaf8386/models/scorer40bb09")

# Import the scorer module
import scorer

# Set the MLflow tracking URI to Unity Catalog
mlflow.set_registry_uri("databricks-uc")

# Set the experiment
mlflow.set_experiment("/Users/benedict@hopsworks.ai/mlpabaf8386/scorer_experiment")

# Start a run
with mlflow.start_run() as run:
    # Log the model
    model_uri = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=scorer
    )
    
    # Register the model in Unity Catalog
    model_name = "workspace.mlpabaf8386.scorer40bb09"
    mv = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )
    
    print(f"Registered model: {model_name}, version: {mv.version}")
