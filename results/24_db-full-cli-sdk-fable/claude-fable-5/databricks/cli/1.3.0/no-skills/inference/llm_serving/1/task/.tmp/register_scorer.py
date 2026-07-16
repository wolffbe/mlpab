# Databricks notebook source
import math

import mlflow
import pandas as pd
from mlflow.models import infer_signature

# model weights (fixed at training time — do not change)
A = 2.449119
B = 1.093524
C = 1.192579
D = 1.949479


def _trigram_weight(tri):
    o0, o1, o2 = (ord(ch) for ch in tri)
    return math.sin(A * o0 + B * o1 + C * o2 + D)


def score(text):
    """Log-likelihood of `text` under the trigram model."""
    ll = 0.0
    for i in range(len(text) - 2):
        ll += _trigram_weight(text[i:i + 3])
    return {"score": round(ll, 6)}


class TrigramScorer(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params=None):
        texts = model_input["text"].tolist()
        return pd.DataFrame([score(t) for t in texts])


# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

example_input = pd.DataFrame({"text": ["hello world"]})
example_output = TrigramScorer().predict(None, example_input)
signature = infer_signature(example_input, example_output)

with mlflow.start_run():
    info = mlflow.pyfunc.log_model(
        name="scorer",
        python_model=TrigramScorer(),
        signature=signature,
        input_example=example_input,
        registered_model_name="workspace.mlpab89d087.scorer83d9cf",
    )

print("registered:", info.registered_model_version)
