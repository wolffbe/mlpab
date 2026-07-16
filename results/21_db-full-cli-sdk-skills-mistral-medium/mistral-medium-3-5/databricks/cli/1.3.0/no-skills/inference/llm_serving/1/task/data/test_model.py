# Databricks notebook source
# MAGIC %md
# MAGIC Test model locally

# COMMAND ----------

import mlflow

# Define the scorer function directly
import json
import math
import sys

A = 2.653901
B = 1.890485
C = 1.091312
D = 1.69396

def _trigram_weight(tri):
    o0, o1, o2 = (ord(ch) for ch in tri)
    return math.sin(A * o0 + B * o1 + C * o2 + D)

def score(text):
    """Log-likelihood of `text` under the trigram model."""
    ll = 0.0
    for i in range(len(text) - 2):
        ll += _trigram_weight(text[i:i + 3])
    return {"score": round(ll, 6)}

# Test the function locally
print("Testing score function:")
print(f"score('hello') = {score('hello')}")
print(f"score('hello world') = {score('hello world')}")

# Create a class that wraps the score function for MLflow
class ScorerModel(mlflow.pyfunc.PythonModel):
    def __init__(self):
        pass
    
    def load_context(self, context):
        pass
    
    def predict(self, context, model_input):
        # Handle different input formats
        if isinstance(model_input, dict):
            # If it's a dict, extract the text
            if "text" in model_input:
                text = model_input["text"]
                return score(text)
            elif "inputs" in model_input:
                # Handle inputs format
                inputs = model_input["inputs"]
                if isinstance(inputs, list) and len(inputs) > 0:
                    return score(inputs[0])
        elif isinstance(model_input, list):
            # If it's a list, use the first element
            if len(model_input) > 0:
                return score(model_input[0])
        elif isinstance(model_input, str):
            # If it's a string, use it directly
            return score(model_input)
        
        # Default
        return {"score": 0.0}

# Test the model locally
model = ScorerModel()
print("\nTesting ScorerModel:")
print(f"model.predict(None, 'hello') = {model.predict(None, 'hello')}")
print(f"model.predict(None, {{'text': 'hello'}}) = {model.predict(None, {'text': 'hello'})}")
print(f"model.predict(None, {{'inputs': ['hello']}}) = {model.predict(None, {'inputs': ['hello']})}")

# Set the MLflow tracking URI to Unity Catalog
mlflow.set_registry_uri("databricks-uc")

# Set the experiment
mlflow.set_experiment("/Users/benedict@hopsworks.ai/mlpabaf8386/scorer_experiment")

# Start a run
with mlflow.start_run() as run:
    # Log the model
    model_info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=ScorerModel()
    )
    
    # Get the model URI as a string
    model_uri = f"runs:/{run.info.run_id}/model"
    
    # Register the model in Unity Catalog
    model_name = "workspace.mlpabaf8386.scorer40bb09"
    mv = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )
    
    print(f"\nRegistered model: {model_name}, version: {mv.version}")
