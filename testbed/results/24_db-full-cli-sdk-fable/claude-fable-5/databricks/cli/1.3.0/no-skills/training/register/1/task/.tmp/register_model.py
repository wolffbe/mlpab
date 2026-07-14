# Databricks notebook source
import json
import math

import mlflow
import pandas as pd
from mlflow import MlflowClient

VOL = "/Volumes/workspace/mlpab0f9e5e/artifacts"
MODEL_NAME = "workspace.mlpab0f9e5e.churnmodel921167"

with open(f"{VOL}/model.json") as f:
    model_spec = json.load(f)
with open(f"{VOL}/metrics.json") as f:
    metrics = json.load(f)

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment("/Users/benedict@hopsworks.ai/mlpab0f9e5e/churn_register")


class ChurnModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        with open(context.artifacts["model_json"]) as fh:
            self.spec = json.load(fh)

    def predict(self, context, model_input):
        w = self.spec["weights"]
        b = self.spec["bias"]
        rows = model_input[self.spec["features"]].values.tolist()
        return [1.0 / (1.0 + math.exp(-(b + sum(wi * xi for wi, xi in zip(w, r))))) for r in rows]


input_example = pd.DataFrame([[0.1, 0.2, 0.3, 0.4, 0.5]], columns=model_spec["features"])

with mlflow.start_run() as run:
    mlflow.log_metrics(metrics)
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=ChurnModel(),
        artifacts={"model_json": f"{VOL}/model.json"},
        input_example=input_example,
    )
    mv = mlflow.register_model(f"runs:/{run.info.run_id}/model", MODEL_NAME)

client = MlflowClient(registry_uri="databricks-uc")
for k, v in metrics.items():
    client.set_registered_model_tag(MODEL_NAME, k, str(v))
    client.set_model_version_tag(MODEL_NAME, mv.version, k, str(v))

print("REGISTERED", MODEL_NAME, "version", mv.version)
