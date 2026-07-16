# Databricks notebook source
import os, shutil, json, tempfile

IN = "/Volumes/workspace/mlpab2b785d/ftvol/input"
OUT = "/Volumes/workspace/mlpab2b785d/ftvol/output"

# --- Run the provided fine-tuning script AS-IS in a clean working dir ---
work = tempfile.mkdtemp(prefix="ftwork_")
os.chdir(work)
for f in ["base_model.npz", "finetune.txt", "eval.txt"]:
    shutil.copy(f"{IN}/{f}", f)

ns = {}
exec(open(f"{IN}/finetune_model.py").read(), ns)
ns["main"]()

metrics = json.load(open("metrics.json"))
print("metrics:", metrics)
assert os.path.exists("finetuned_model.npz")

# --- Persist outputs back to the volume ---
os.makedirs(OUT, exist_ok=True)
for f in ["finetuned_model.npz", "metrics.json"]:
    shutil.copy(f, f"{OUT}/{f}")
print("outputs written to", OUT)

# COMMAND ----------

# --- Register the fine-tuned model in the Unity Catalog model registry ---
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

SIGNATURE = ModelSignature(
    inputs=Schema([ColSpec("string", "input")]),
    outputs=Schema([ColSpec("string", "output")]),
)

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment("/Users/benedict@logicalclocks.com/mlpab2b785d/ftexp698f06")

MODEL_NAME = "workspace.mlpab2b785d.ftmodel698f06"


class NpzModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import numpy as np
        self._d = dict(np.load(context.artifacts["npz"], allow_pickle=True))

    def predict(self, context, model_input):
        return None


with mlflow.start_run() as run:
    mlflow.log_metric("eval_loss", float(metrics["eval_loss"]))
    mlflow.log_metric("base_eval_loss", float(metrics["base_eval_loss"]))
    mlflow.set_tags({
        "eval_loss": metrics["eval_loss"],
        "base_eval_loss": metrics["base_eval_loss"],
    })
    info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=NpzModel(),
        artifacts={"npz": f"{OUT}/finetuned_model.npz"},
        signature=SIGNATURE,
    )

mv = mlflow.register_model(model_uri=info.model_uri, name=MODEL_NAME)
client = MlflowClient(registry_uri="databricks-uc")
client.set_model_version_tag(MODEL_NAME, mv.version, "eval_loss", str(metrics["eval_loss"]))
client.set_model_version_tag(MODEL_NAME, mv.version, "base_eval_loss", str(metrics["base_eval_loss"]))
client.update_model_version(
    MODEL_NAME, mv.version,
    description=f"eval_loss={metrics['eval_loss']}, base_eval_loss={metrics['base_eval_loss']}",
)
print("registered", MODEL_NAME, "version", mv.version)

# COMMAND ----------

dbutils.notebook.exit(json.dumps(metrics))
