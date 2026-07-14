#!/usr/bin/env python3
"""
Script to package and register the scorer model.
"""
import os
import shutil
import mlflow
from mlflow.pyfunc import PythonModel
from scorer import score

# Set the tracking URI to use Databricks
mlflow.set_registry_uri("databricks")

# Create a temporary directory for the model
model_dir = ".tmp/model"
os.makedirs(model_dir, exist_ok=True)

# Copy the scorer.py file to the model directory
shutil.copy("data/scorer.py", f"{model_dir}/scorer.py")

# Create a wrapper class that includes the scorer function
class ScorerModel(PythonModel):
    def load_context(self, context):
        from scorer import score
        self.score = score
    
    def predict(self, context, model_input):
        return self.score(model_input)

# Create an MLflow model
with mlflow.start_run():
    # Log the model
    mlflow.pyfunc.log_model(
        artifact_path="scorer_model",
        python_model=ScorerModel(),
        artifacts={"scorer": f"{model_dir}/scorer.py"},
        code_path=[f"{model_dir}/scorer.py"],
        registered_model_name=f"{os.environ['MLPAB_DATABRICKS_SCHEMA']}.scorer_model"
    )