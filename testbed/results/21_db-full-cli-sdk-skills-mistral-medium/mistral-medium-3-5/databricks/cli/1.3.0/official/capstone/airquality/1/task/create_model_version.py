# Databricks notebook source
# MAGIC %md
# MAGIC ## Create Model Version for airqmodelb70a96

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import ModelVersionInfo
import json

# COMMAND ----------

w = WorkspaceClient()

catalog_name = "workspace"
schema_name = "mlpab8a96ff"
model_name = "airqmodelb70a96"

# COMMAND ----------

# Get the registered model
model = w.registered_models.get(f"{catalog_name}.{schema_name}.{model_name}")
print(f"Model: {model}")

# COMMAND ----------

# The model version needs to point to a storage location with model artifacts
# We have the model saved at /Users/benedict@hopsworks.ai/mlpab8a96ff/airqmodelb70a96.joblib
# But we need to upload it to the model's storage location

# First, let's check the model's storage location
storage_location = model.storage_location
print(f"Storage location: {storage_location}")

# COMMAND ----------

# Create a model version
# We need to specify the source path where the model artifacts are stored
# The source should be a URI pointing to the model artifacts

# For Unity Catalog models, the source should be in the format:
# dbfs:/path/to/model or s3://path/to/model

# Let's try to create a model version
try:
    model_version = w.model_versions.create(
        name=f"{catalog_name}.{schema_name}.{model_name}",
        comment="Initial version with RandomForest regressor",
        # The source needs to be a URI to the model artifacts
        # We'll use the workspace path
    )
    print(f"Created model version: {model_version}")
except Exception as e:
    print(f"Error creating model version: {e}")

# COMMAND ----------
