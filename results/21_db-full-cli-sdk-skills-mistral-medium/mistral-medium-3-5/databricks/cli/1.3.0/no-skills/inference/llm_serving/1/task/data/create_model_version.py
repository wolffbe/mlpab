# Databricks notebook source
# MAGIC %md
# MAGIC Create model version for scorer40bb09

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient

# Set the MLflow tracking URI
mlflow.set_registry_uri("databricks-uc")

# Create MLflow client
client = MlflowClient()

# Create a model version
model_name = "workspace.mlpabaf8386.scorer40bb09"
source = "dbfs:/Volumes/workspace/mlpabaf8386/models/scorer40bb09"

# Create the model version
mv = client.create_model_version(
    name=model_name,
    source=source,
    run_id=None
)

print(f"Created model version: {mv.version}")
