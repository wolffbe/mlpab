# Databricks notebook source
import os, sys, shutil, subprocess, json, tempfile

VOL = "/Volumes/workspace/mlpabd475ab/ftvol"
INPUTS = VOL + "/inputs"
OUTPUTS = VOL + "/outputs"
WORK = tempfile.mkdtemp(prefix="ftwork_")

os.makedirs(OUTPUTS, exist_ok=True)

# Stage the provided script and data into the working dir, unmodified.
for f in ["base_model.npz", "finetune.txt", "eval.txt", "finetune_model.py"]:
    shutil.copy(f"{INPUTS}/{f}", f"{WORK}/{f}")

# Run the provided fine-tuning script exactly as-is (as __main__).
res = subprocess.run([sys.executable, "finetune_model.py"], cwd=WORK,
                     capture_output=True, text=True)
print("STDOUT:", res.stdout)
print("STDERR:", res.stderr)
assert res.returncode == 0, f"finetune script failed rc={res.returncode}"

# Read outputs produced by the script.
with open(f"{WORK}/metrics.json") as fh:
    metrics = json.load(fh)
print("metrics:", metrics)
eval_loss = metrics["eval_loss"]
base_eval_loss = metrics["base_eval_loss"]

# Persist outputs back to the volume.
shutil.copy(f"{WORK}/finetuned_model.npz", f"{OUTPUTS}/finetuned_model.npz")
shutil.copy(f"{WORK}/metrics.json", f"{OUTPUTS}/metrics.json")

# COMMAND ----------

# Register the fine-tuned model in the Unity Catalog model registry.
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment("/Users/benedict@logicalclocks.com/mlpabd475ab/ftexp")

MODEL_NAME = "workspace.mlpabd475ab.ftmodel698f06"
npz_path = f"{WORK}/finetuned_model.npz"


class FineTunedModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import numpy as np
        self.ckpt = np.load(context.artifacts["npz"])

    def predict(self, context, model_input):
        import numpy as np
        logits = self.ckpt["logits"]
        return logits


from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
import numpy as np

signature = ModelSignature(
    inputs=Schema([TensorSpec(np.dtype("float64"), (-1,))]),
    outputs=Schema([TensorSpec(np.dtype("float64"), (-1, -1))]),
)

with mlflow.start_run() as run:
    mlflow.log_metric("eval_loss", eval_loss)
    mlflow.log_metric("base_eval_loss", base_eval_loss)
    info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=FineTunedModel(),
        artifacts={"npz": npz_path},
        signature=signature,
        registered_model_name=MODEL_NAME,
    )
    run_id = run.info.run_id
print("logged & registered, run_id:", run_id)

# Attach metrics to the registry entry (model version tags).
client = MlflowClient(registry_uri="databricks-uc")
# find the version just created
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
version = max(int(v.version) for v in versions)
print("registered version:", version)
client.set_model_version_tag(MODEL_NAME, str(version), "eval_loss", str(eval_loss))
client.set_model_version_tag(MODEL_NAME, str(version), "base_eval_loss", str(base_eval_loss))
client.update_model_version(MODEL_NAME, str(version),
    description=f"Fine-tuned bigram model. eval_loss={eval_loss}, base_eval_loss={base_eval_loss}")

print(json.dumps({"model_name": "ftmodel698f06", "version": version,
                  "eval_loss": eval_loss, "base_eval_loss": base_eval_loss}))
