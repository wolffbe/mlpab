# Databricks notebook source
# MAGIC %md
# MAGIC ## Create MLflow Model for Scorer

import mlflow
import os
import shutil

# Create a temporary directory for the model
temp_dir = "/tmp/scorer_model"
if os.path.exists(temp_dir):
    shutil.rmtree(temp_dir)
os.makedirs(temp_dir)

# Copy the scorer.py file to the model directory
scorer_code = '''
"""A tiny deterministic pure-python "language model".

Character-trigram log-likelihood scorer: every trigram of the input text
contributes a weight derived from fixed constants; the score is the sum,
rounded to 6 decimal places. No dependencies beyond the standard library;
fully deterministic — the same text always yields the same score.

    >>> from scorer import score
    >>> score("hello world")
    {"score": ...}
"""
import json
import math
import sys

# model weights (fixed at training time — do not change)
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


def _load_pyfunc(data_path):
    import scorer
    return scorer


def predict(context, model_input):
    import json
    result = model_input
    if isinstance(model_input, dict):
        if "inputs" in model_input:
            result = model_input["inputs"]
        elif "data" in model_input:
            result = model_input["data"]
    if isinstance(result, list):
        result = result[0]
    if isinstance(result, str):
        return score(result)
    return {"score": 0.0}
'''

# Write the scorer module
with open(os.path.join(temp_dir, "scorer.py"), "w") as f:
    f.write(scorer_code)

# Create MLmodel file
mlmodel_content = '''signature:
  inputs: '[{"name": "text", "type": "string"}]'
  outputs: '[{"name": "score", "type": "double"}]'

loader_module: "mlflow.pyfunc"

pyfunc:
  loader: _load_pyfunc
  predict_fn: predict
'''

with open(os.path.join(temp_dir, "MLmodel"), "w") as f:
    f.write(mlmodel_content)

# Create conda.yaml
conda_content = '''name: scorer-env
channels:
  - conda-forge
dependencies:
  - python=3.9
  - pip
  - pip:
    - mlflow
'''

with open(os.path.join(temp_dir, "conda.yaml"), "w") as f:
    f.write(conda_content)

# Log the model with MLflow
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment("/Users/benedict@hopsworks.ai/mlpab6a1230/scorer_experiment")

with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        artifact_path="scorer_model",
        python_model=mlflow.pyfunc.PythonModel(
            loader_module="scorer",
            predict_fn=predict
        ),
        registered_model_name="workspace.mlpab6a1230.scorer40bb09"
    )

print("Model created and registered successfully!")
