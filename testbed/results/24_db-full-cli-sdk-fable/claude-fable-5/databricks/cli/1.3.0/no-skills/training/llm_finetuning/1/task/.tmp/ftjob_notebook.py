# Databricks notebook source
import json
import os
import shutil
import subprocess
import sys
import tempfile

VOL = "/Volumes/workspace/mlpabbadb45/ftjob79b056_vol"

workdir = tempfile.mkdtemp()
for f in ["base_model.npz", "eval.txt", "finetune.txt", "finetune_model.py"]:
    shutil.copy(os.path.join(VOL, f), os.path.join(workdir, f))

result = subprocess.run(
    [sys.executable, "finetune_model.py"],
    cwd=workdir,
    capture_output=True,
    text=True,
)
print(result.stdout)
print(result.stderr)
result.check_returncode()

metrics = json.load(open(os.path.join(workdir, "metrics.json")))
print("metrics:", metrics)

# Copy outputs back to the volume for readback
shutil.copy(os.path.join(workdir, "finetuned_model.npz"), os.path.join(VOL, "finetuned_model.npz"))
shutil.copy(os.path.join(workdir, "metrics.json"), os.path.join(VOL, "metrics.json"))

# COMMAND ----------

import mlflow

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment("/Users/benedict@hopsworks.ai/mlpabbadb45/ftjob79b056_experiment")

MODEL_NAME = "workspace.mlpabbadb45.ftmodel79b056"


class FinetunedBigramModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import numpy as np

        ckpt = np.load(context.artifacts["finetuned_model"])
        self.logits = ckpt["logits"]
        self.vocab = [str(ch) for ch in ckpt["vocab"]] if "vocab" in ckpt else None

    def predict(self, context, model_input):
        return model_input


from mlflow.models import infer_signature
import pandas as pd

example = pd.DataFrame({"text": ["a"]})
signature = infer_signature(example, example)

with mlflow.start_run(run_name="ftjob79b056") as run:
    mlflow.log_metrics(metrics)
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=FinetunedBigramModel(),
        artifacts={"finetuned_model": os.path.join(workdir, "finetuned_model.npz")},
        signature=signature,
        input_example=example,
        registered_model_name=MODEL_NAME,
    )

from mlflow.tracking import MlflowClient

client = MlflowClient()
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
version = sorted(int(v.version) for v in versions)[-1]
print("registered version:", version)

desc = json.dumps(metrics)
client.update_registered_model(MODEL_NAME, description="Fine-tuned bigram model. metrics: " + desc)
client.update_model_version(MODEL_NAME, str(version), description="metrics: " + desc)
for k, v in metrics.items():
    client.set_registered_model_tag(MODEL_NAME, k, str(v))
    client.set_model_version_tag(MODEL_NAME, str(version), k, str(v))

print("done", metrics)
