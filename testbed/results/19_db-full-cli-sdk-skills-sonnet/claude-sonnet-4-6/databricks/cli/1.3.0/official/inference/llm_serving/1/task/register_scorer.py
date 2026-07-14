# Databricks notebook source
import mlflow
import mlflow.pyfunc
import math
import pandas as pd

# scorer constants (same as data/scorer.py)
A = 2.158311
B = 2.825397
C = 1.859128
D = -0.317766


def _trigram_weight(tri):
    o0, o1, o2 = (ord(ch) for ch in tri)
    return math.sin(A * o0 + B * o1 + C * o2 + D)


def score(text):
    ll = 0.0
    for i in range(len(text) - 2):
        ll += _trigram_weight(text[i:i + 3])
    return {"score": round(ll, 6)}


class ScorerModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params=None):
        if isinstance(model_input, pd.DataFrame):
            texts = model_input.iloc[:, 0].tolist()
        elif isinstance(model_input, list):
            texts = model_input
        else:
            texts = [str(model_input)]
        results = []
        for text in texts:
            results.append(score(str(text)))
        return results


CATALOG = "workspace"
SCHEMA = "mlpab2863d5"
MODEL_NAME = "scorer98c848"
full_model_name = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(f"/Users/benedict@logicalclocks.com/mlpab2863d5/scorer_experiment")

with mlflow.start_run():
    signature = mlflow.models.infer_signature(
        pd.DataFrame({"text": ["hello world"]}),
        [{"score": 0.0}]
    )
    model_info = mlflow.pyfunc.log_model(
        artifact_path="scorer_model",
        python_model=ScorerModel(),
        registered_model_name=full_model_name,
        signature=signature,
        input_example=pd.DataFrame({"text": ["hello world"]})
    )

print(f"Registered model: {full_model_name}")
print(f"Model URI: {model_info.model_uri}")

# Get the latest version
client = mlflow.tracking.MlflowClient(registry_uri="databricks-uc")
versions = client.search_model_versions(f"name='{full_model_name}'")
latest = max(versions, key=lambda v: int(v.version))
print(f"Latest version: {latest.version}")
