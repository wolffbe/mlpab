# Databricks notebook source
# MAGIC %pip install mlflow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import mlflow.pyfunc
import json, os, math

mlflow.set_registry_uri("databricks-uc")

# The artifact file content (data/model.json) embedded verbatim as the model's content
MODEL_JSON = json.dumps({
    "model_type": "logistic_regression",
    "features": ["f1", "f2", "f3", "f4", "f5"],
    "weights": [-0.753078, 1.426403, -0.652302, 1.163791, 0.829019],
    "bias": 0.129202,
}, indent=2)

METRICS = {"auc": 0.8524, "accuracy": 0.7254}

MODEL_NAME = "workspace.mlpabbe5553.churnmodel7948d0"
EXPERIMENT = "/Users/benedict@logicalclocks.com/mlpabbe5553/churn_register_exp"

# Write the artifact file on the driver so it can be logged as the model's content
art_dir = "/tmp/churn_artifacts"
os.makedirs(art_dir, exist_ok=True)
art_path = os.path.join(art_dir, "model.json")
with open(art_path, "w") as f:
    f.write(MODEL_JSON)


class ChurnModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        with open(context.artifacts["model_json"]) as fh:
            self.m = json.load(fh)

    def predict(self, context, model_input):
        w = self.m["weights"]
        b = self.m["bias"]
        feats = self.m["features"]
        out = []
        for _, row in model_input.iterrows():
            z = b + sum(w[i] * float(row[feats[i]]) for i in range(len(w)))
            out.append(1.0 / (1.0 + math.exp(-z)))
        return out


from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec
import pandas as pd

input_schema = Schema([ColSpec("double", f) for f in ["f1", "f2", "f3", "f4", "f5"]])
output_schema = Schema([ColSpec("double")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)
input_example = pd.DataFrame([{ "f1": 0.1, "f2": 0.2, "f3": 0.3, "f4": 0.4, "f5": 0.5 }])

mlflow.set_experiment(EXPERIMENT)

with mlflow.start_run() as run:
    mlflow.log_metrics(METRICS)
    # Log the raw artifact file directly so its content is part of the model
    mlflow.log_artifact(art_path)
    info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=ChurnModel(),
        artifacts={"model_json": art_path},
        signature=signature,
        input_example=input_example,
        registered_model_name=MODEL_NAME,
    )
    run_id = run.info.run_id

print("REGISTERED_MODEL", MODEL_NAME)
print("RUN_ID", run_id)

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient(registry_uri="databricks-uc")
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
ver = None
for v in versions:
    if v.run_id == run_id:
        ver = v.version
        break
if ver is None and versions:
    ver = max(int(v.version) for v in versions)
print("MODEL_VERSION", ver)

# Attach metrics as model-version tags too for robust readback
for k, val in METRICS.items():
    client.set_model_version_tag(MODEL_NAME, str(ver), f"metric.{k}", str(val))

print("DONE")
