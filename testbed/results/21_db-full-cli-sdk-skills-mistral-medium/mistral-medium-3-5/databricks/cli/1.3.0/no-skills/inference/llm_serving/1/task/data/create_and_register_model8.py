# Databricks notebook source
# MAGIC %md
# MAGIC Create and register model for scorer40bb09

# COMMAND ----------

import mlflow
from mlflow.models import infer_signature

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

# Set the MLflow tracking URI to Unity Catalog
mlflow.set_registry_uri("databricks-uc")

# Set the experiment
mlflow.set_experiment("/Users/benedict@hopsworks.ai/mlpabaf8386/scorer_experiment")

# Start a run
with mlflow.start_run() as run:
    # Infer the signature
    signature = infer_signature("hello world", score("hello world"))
    
    # Log the model with signature
    model_uri = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=score,
        signature=signature,
        input_example={"text": "hello world"}
    )
    
    # Register the model in Unity Catalog
    model_name = "workspace.mlpabaf8386.scorer40bb09"
    mv = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )
    
    print(f"Registered model: {model_name}, version: {mv.version}")
