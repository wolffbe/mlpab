# Databricks notebook source
# MAGIC %python

import mlflow
import mlflow.pyfunc
import math
import json
import pandas as pd

# ── constants ───────────────────────────────────────────────────────────────
SCHEMA     = "workspace.mlpab10b900"
MODEL_NAME = f"{SCHEMA}.scorer98c848"
EXPERIMENT = "/Users/benedict@logicalclocks.com/mlpab10b900/scorer"

# ── model definition ─────────────────────────────────────────────────────────
class ScorerModel(mlflow.pyfunc.PythonModel):

    A = 2.158311
    B = 2.825397
    C = 1.859128
    D = -0.317766

    def _trigram_weight(self, tri):
        o0, o1, o2 = (ord(ch) for ch in tri)
        return math.sin(self.A * o0 + self.B * o1 + self.C * o2 + self.D)

    def _score(self, text):
        ll = 0.0
        for i in range(len(text) - 2):
            ll += self._trigram_weight(text[i:i + 3])
        return round(ll, 6)

    def predict(self, context, model_input, params=None):
        # model_input is a pandas DataFrame with one column "text"
        if isinstance(model_input, pd.DataFrame):
            col = model_input.columns[0]
            return [{"score": self._score(str(row))} for row in model_input[col]]
        if isinstance(model_input, list):
            return [{"score": self._score(str(t))} for t in model_input]
        return {"score": self._score(str(model_input))}


# ── signature ────────────────────────────────────────────────────────────────
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

input_schema  = Schema([ColSpec("string", "text")])
output_schema = Schema([ColSpec("double", "score")])
signature     = ModelSignature(inputs=input_schema, outputs=output_schema)

# ── log & register ───────────────────────────────────────────────────────────
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT)

with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        artifact_path="scorer",
        python_model=ScorerModel(),
        signature=signature,
        registered_model_name=MODEL_NAME,
    )
    run_id = run.info.run_id

print(f"run_id={run_id}")
print(f"model={MODEL_NAME}")

# get latest version
client = mlflow.tracking.MlflowClient()
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
latest = max(int(v.version) for v in versions)
print(f"version={latest}")
