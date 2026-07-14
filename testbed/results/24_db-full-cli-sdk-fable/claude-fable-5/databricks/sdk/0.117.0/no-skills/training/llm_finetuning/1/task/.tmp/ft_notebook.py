# Databricks notebook source
import json
import os
import runpy
import shutil
import tempfile

VOL = "/Volumes/workspace/mlpab05c114/ftvol79b056"
MODEL_NAME = "workspace.mlpab05c114.ftmodel79b056"
EXPERIMENT = "/Users/benedict@hopsworks.ai/mlpab05c114/ftjob79b056_exp"

work = tempfile.mkdtemp()
for f in ["base_model.npz", "finetune.txt", "eval.txt", "finetune_model.py"]:
    shutil.copy(os.path.join(VOL, f), os.path.join(work, f))
os.chdir(work)

# Run the provided fine-tuning script exactly as-is, as __main__.
runpy.run_path("finetune_model.py", run_name="__main__")

shutil.copy("finetuned_model.npz", os.path.join(VOL, "finetuned_model.npz"))
shutil.copy("metrics.json", os.path.join(VOL, "metrics.json"))
metrics = json.load(open("metrics.json"))
print("METRICS:", metrics)

# COMMAND ----------

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT)


class BigramAdapterModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import numpy as np

        ckpt = np.load(context.artifacts["finetuned_model"])
        self.logits = ckpt["logits"]

    def predict(self, context, model_input):
        return model_input


input_example = pd.DataFrame({"text": ["a"]})

with mlflow.start_run(run_name="ftjob79b056") as run:
    mlflow.log_metrics(metrics)
    info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=BigramAdapterModel(),
        artifacts={"finetuned_model": os.path.join(work, "finetuned_model.npz")},
        input_example=input_example,
        registered_model_name=MODEL_NAME,
    )

client = MlflowClient(registry_uri="databricks-uc")
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
latest = max(int(v.version) for v in versions)
for k, val in metrics.items():
    client.set_model_version_tag(MODEL_NAME, str(latest), k, str(val))
    client.set_registered_model_tag(MODEL_NAME, k, str(val))
print("REGISTERED:", MODEL_NAME, "version", latest)
print("FINAL:", json.dumps({"metrics": metrics, "version": latest}))
