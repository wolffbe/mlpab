# Databricks notebook source
# MAGIC %md
# MAGIC ## Register Scorer Model

# COMMAND ----------

import mlflow
import sys
import os
from pyspark.sql import SparkSession

# Add the current directory to Python path to import scorer
sys.path.append(os.getcwd())
from scorer import score

# Set the tracking URI to use Databricks
mlflow.set_registry_uri("databricks")

# Create a wrapper class for the scorer function
class ScorerModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        pass
    
    def predict(self, context, model_input):
        return score(model_input)

# Create a model in the specified schema
schema_name = os.environ["MLPAB_DATABRICKS_SCHEMA"]
model_name = f"{schema_name}.scorer_model"

# Log the model
with mlflow.start_run():
    mlflow.pyfunc.log_model(
        artifact_path="scorer_model",
        python_model=ScorerModel(),
        registered_model_name=model_name
    )

print(f"Model registered as: {model_name}")