# Databricks notebook source
# Register the fine-tuned model into the Unity Catalog model registry as
# workspace.mlpab5b087a.ftmodel698f06 version 1 using MLflow (the platform's UC
# registry client). The fine-tuned npz is the model content; metrics.json values
# are attached as run metrics and model-version tags. Runs on serverless.
import json, shutil, tempfile, os
import numpy as np
import mlflow
from mlflow.pyfunc import PythonModel
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

STAGE = "/Volumes/workspace/mlpab5b087a/ftstage"
FULL = "workspace.mlpab5b087a.ftmodel698f06"
USER = "benedict@logicalclocks.com"

metrics = json.load(open(f"{STAGE}/metrics.json"))
work = tempfile.mkdtemp(prefix="reg_")
npz = os.path.join(work, "finetuned_model.npz")
shutil.copy(f"{STAGE}/finetuned_model.npz", npz)


class FineTunedBigram(PythonModel):
    def load_context(self, context):
        import numpy as np
        ck = np.load(context.artifacts["model_npz"], allow_pickle=True)
        self.logits = ck["logits"]

    def predict(self, context, model_input):
        import numpy as np
        z = self.logits - self.logits.max(axis=1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(axis=1, keepdims=True)


mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(f"/Users/{USER}/mlpab5b087a/ftmodel698f06_exp")

signature = infer_signature(
    np.zeros((1, 1), dtype="float64"),
    np.zeros((2, 2), dtype="float64"),
)

with mlflow.start_run(run_name="ftmodel698f06_v1") as run:
    mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=FineTunedBigram(),
        artifacts={"model_npz": npz},
        signature=signature,
        registered_model_name=FULL,
        pip_requirements=["numpy"],
    )
    run_id = run.info.run_id

client = MlflowClient(registry_uri="databricks-uc")
versions = [int(v.version) for v in client.search_model_versions(f"name='{FULL}'")]
version = max(versions)
for k, v in metrics.items():
    client.set_model_version_tag(FULL, str(version), k, str(v))
try:
    client.set_registered_model_alias(FULL, "prod", version)
except Exception:
    pass

dbutils.notebook.exit(json.dumps({"version": version, "run_id": run_id,
                                  "all_versions": versions, "metrics": metrics}))
