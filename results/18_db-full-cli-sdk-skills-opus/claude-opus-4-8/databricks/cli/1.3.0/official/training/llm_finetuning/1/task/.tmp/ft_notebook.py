# Databricks notebook source
# Wrapper notebook: runs the provided finetune_model.py AS-IS as a platform job,
# then registers the fine-tuned model in Unity Catalog model registry.

import os
import shutil
import json
import runpy

IN = "/Volumes/workspace/mlpab8a19e8/ftstage/input"
OUT = "/Volumes/workspace/mlpab8a19e8/ftstage/output"
WORK = "/tmp/ftwork"
UC_MODEL = "workspace.mlpab8a19e8.ftmodel698f06"

os.makedirs(WORK, exist_ok=True)
os.makedirs(OUT, exist_ok=True)
for f in ["base_model.npz", "finetune.txt", "eval.txt", "finetune_model.py"]:
    shutil.copy(os.path.join(IN, f), os.path.join(WORK, f))

os.chdir(WORK)

# Run the provided script EXACTLY as-is (it has __name__ == "__main__" guard).
runpy.run_path(os.path.join(WORK, "finetune_model.py"), run_name="__main__")

# Read outputs produced by the script.
with open(os.path.join(WORK, "metrics.json")) as fh:
    metrics = json.load(fh)
print("metrics:", metrics)

# Persist outputs back to the volume so they can be read after the job.
shutil.copy(os.path.join(WORK, "finetuned_model.npz"), os.path.join(OUT, "finetuned_model.npz"))
shutil.copy(os.path.join(WORK, "metrics.json"), os.path.join(OUT, "metrics.json"))

# COMMAND ----------

# Register the fine-tuned model in the Unity Catalog model registry.
import numpy as np
import mlflow
from mlflow.pyfunc import PythonModel
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment("/Users/benedict@logicalclocks.com/mlpab8a19e8/ftexp")


class NpzModel(PythonModel):
    def load_context(self, context):
        import numpy as np
        self.ckpt = dict(np.load(context.artifacts["model_file"], allow_pickle=True))

    def predict(self, context, model_input):
        return self.ckpt["logits"]


eval_loss = float(metrics["eval_loss"])
base_eval_loss = float(metrics["base_eval_loss"])

# Build a signature so the model is registrable in Unity Catalog.
_ckpt = np.load(os.path.join(WORK, "finetuned_model.npz"))
_logits = _ckpt["logits"]
_input_example = np.zeros((1, 1), dtype=np.float64)
_signature = infer_signature(_input_example, _logits)

with mlflow.start_run() as run:
    mlflow.log_metrics({"eval_loss": eval_loss, "base_eval_loss": base_eval_loss})
    info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=NpzModel(),
        artifacts={"model_file": os.path.join(WORK, "finetuned_model.npz")},
        signature=_signature,
        input_example=_input_example,
    )

result = mlflow.register_model(model_uri=info.model_uri, name=UC_MODEL)
version = result.version
print("registered version:", version)

client = MlflowClient()
client.set_model_version_tag(UC_MODEL, version, "eval_loss", str(eval_loss))
client.set_model_version_tag(UC_MODEL, version, "base_eval_loss", str(base_eval_loss))

# Surface results as job output.
dbutils.notebook.exit(json.dumps({
    "eval_loss": eval_loss,
    "base_eval_loss": base_eval_loss,
    "model_name": UC_MODEL,
    "version": version,
}))
