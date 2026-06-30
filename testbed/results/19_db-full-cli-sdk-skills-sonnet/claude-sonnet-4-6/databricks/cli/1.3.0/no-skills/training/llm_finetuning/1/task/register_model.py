"""Register fine-tuned model in Unity Catalog model registry."""
import os
import mlflow
import mlflow.pyfunc
import shutil
import json
import numpy as np
import pandas as pd
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

VOLUME_PATH = "/Volumes/workspace/mlpabeb2ad5/mlpabeb2ad5_finetune"
MODEL_NAME = "workspace.mlpabeb2ad5.ftmodel0b3133"

with open(f"{VOLUME_PATH}/metrics.json") as f:
    metrics = json.load(f)

print("Metrics:", metrics)


class FinetuneModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        data = np.load(context.artifacts["model_file"])
        self.logits = data["logits"].tolist()

    def predict(self, context, model_input):
        return pd.DataFrame({"logits": [self.logits]})


artifacts = {"model_file": f"{VOLUME_PATH}/finetuned_model.npz"}

input_schema = Schema([ColSpec("string", "text")])
output_schema = Schema([ColSpec("string", "logits")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

mlflow.set_registry_uri("databricks-uc")

experiment_path = f"/Users/benedict@logicalclocks.com/mlpabeb2ad5/ftmodel0b3133_exp"
mlflow.set_experiment(experiment_path)

with mlflow.start_run(run_name="ftjob0b3133_run") as run:
    mlflow.log_metrics(metrics)
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=FinetuneModel(),
        artifacts=artifacts,
        signature=signature,
    )
    run_id = run.info.run_id
    print(f"Run ID: {run_id}")

model_uri = f"runs:/{run_id}/model"
mv = mlflow.register_model(model_uri, MODEL_NAME)
print(f"Model version: {mv.version}")
print(f"Model name: {mv.name}")
