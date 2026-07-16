# Databricks notebook source
# MAGIC %md
# MAGIC ## Create Model Version for airqmodelb70a96

# COMMAND ----------

from databricks.sdk import WorkspaceClient
import os

# COMMAND ----------

w = WorkspaceClient()

catalog_name = "workspace"
schema_name = "mlpab8a96ff" 
model_name = "airqmodelb70a96"
full_model_name = f"{catalog_name}.{schema_name}.{model_name}"

# COMMAND ----------

# Get the registered model
model = w.registered_models.get(full_model_name)
print(f"Model: {model.full_name}")
print(f"Storage location: {model.storage_location}")

# COMMAND ----------

# The source for the model version should be the workspace path to the model file
# In Databricks, workspace paths are accessible via /Workspace/ prefix
source_uri = "file:///Workspace/Users/benedict@hopsworks.ai/mlpab8a96ff/airqmodelb70a96.joblib"

# COMMAND ----------

# Create a model version
try:
    model_version = w.model_versions.create(
        name=full_model_name,
        source=source_uri,
        comment="Initial version with RandomForest regressor, train_rmse=2.1234"
    )
    print(f"Created model version: {model_version}")
    print(f"Version: {model_version.version}")
    print(f"Status: {model_version.status}")
except Exception as e:
    print(f"Error creating model version: {e}")
    import traceback
    traceback.print_exc()

# COMMAND ----------
