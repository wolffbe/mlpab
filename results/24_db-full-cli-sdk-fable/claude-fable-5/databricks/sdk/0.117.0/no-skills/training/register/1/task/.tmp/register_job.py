"""Runs ON Databricks (serverless job): registers the churn model in UC with metrics."""
import json

import mlflow
from mlflow.models import ModelSignature
from mlflow.tracking import MlflowClient
from mlflow.types.schema import ColSpec, Schema

VOL = "/Volumes/workspace/mlpab3d22c1/artifacts"
FULL_NAME = "workspace.mlpab3d22c1.churnmodel921167"
EXPERIMENT = "/Users/benedict@hopsworks.ai/mlpab3d22c1/churn-register"

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT)

with open(f"{VOL}/metrics.json") as f:
    metrics = json.load(f)
with open(f"{VOL}/model.json") as f:
    model_def = json.load(f)


class ChurnModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import json as _json
        with open(context.artifacts["model"]) as fh:
            self._m = _json.load(fh)

    def predict(self, context, model_input, params=None):
        import math
        w = self._m["weights"]
        b = self._m["bias"]
        feats = self._m["features"]
        rows = model_input[feats].values.tolist()
        return [
            1.0 / (1.0 + math.exp(-(sum(wi * xi for wi, xi in zip(w, row)) + b)))
            for row in rows
        ]


signature = ModelSignature(
    inputs=Schema([ColSpec("double", f) for f in model_def["features"]]),
    outputs=Schema([ColSpec("double")]),
)

with mlflow.start_run(run_name="register-churnmodel921167") as run:
    mlflow.log_metrics(metrics)
    info = mlflow.pyfunc.log_model(
        "model",
        python_model=ChurnModel(),
        artifacts={"model": f"{VOL}/model.json"},
        signature=signature,
        registered_model_name=FULL_NAME,
    )

version = info.registered_model_version
client = MlflowClient(registry_uri="databricks-uc")
for k, v in metrics.items():
    client.set_model_version_tag(FULL_NAME, version, k, str(v))
    client.set_registered_model_tag(FULL_NAME, k, str(v))
client.update_model_version(
    FULL_NAME, version, description=f"Churn model. Metrics: {json.dumps(metrics)}"
)

print("REGISTERED", FULL_NAME, "version", version, "run", run.info.run_id)
