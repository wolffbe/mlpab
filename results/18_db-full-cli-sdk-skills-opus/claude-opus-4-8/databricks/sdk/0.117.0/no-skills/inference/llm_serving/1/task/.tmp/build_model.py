# Databricks notebook source
# MAGIC %pip install mlflow pandas

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# Logs the deterministic trigram scorer as an MLflow pyfunc model and
# registers it to Unity Catalog so it can be served by Model Serving.

import math
import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow.models.signature import infer_signature

mlflow.set_registry_uri("databricks-uc")

EXP_PATH = "/Users/benedict@logicalclocks.com/mlpab695e17/scorer_exp"
mlflow.set_experiment(EXP_PATH)

UC_MODEL_NAME = "workspace.mlpab695e17.scorer78e1bc"

# model weights (fixed at training time — do not change)
A = 2.986326
B = 2.292444
C = 1.035017
D = -1.740277


def _score_text(text):
    ll = 0.0
    for i in range(len(text) - 2):
        tri = text[i:i + 3]
        o0, o1, o2 = ord(tri[0]), ord(tri[1]), ord(tri[2])
        ll += math.sin(A * o0 + B * o1 + C * o2 + D)
    return round(ll, 6)


class ScorerModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input):
        import math
        import pandas as pd
        A = 2.986326
        B = 2.292444
        C = 1.035017
        D = -1.740277

        def score_text(text):
            ll = 0.0
            for i in range(len(text) - 2):
                tri = text[i:i + 3]
                o0, o1, o2 = ord(tri[0]), ord(tri[1]), ord(tri[2])
                ll += math.sin(A * o0 + B * o1 + C * o2 + D)
            return round(ll, 6)

        if isinstance(model_input, pd.DataFrame):
            if "text" in model_input.columns:
                texts = model_input["text"].tolist()
            else:
                texts = model_input.iloc[:, 0].tolist()
        elif isinstance(model_input, dict):
            texts = list(model_input.get("text", []))
        else:
            texts = list(model_input)
        return [score_text(str(t)) for t in texts]


input_example = pd.DataFrame({"text": ["hello world"]})
output_example = [_score_text("hello world")]
signature = infer_signature(input_example, output_example)

with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=ScorerModel(),
        signature=signature,
        input_example=input_example,
        registered_model_name=UC_MODEL_NAME,
        pip_requirements=["mlflow", "pandas"],
    )
    print("RUN_ID", run.info.run_id)

from mlflow.tracking import MlflowClient
client = MlflowClient(registry_uri="databricks-uc")
versions = client.search_model_versions("name='%s'" % UC_MODEL_NAME)
latest = max(int(v.version) for v in versions)
print("REGISTERED_MODEL", UC_MODEL_NAME, "VERSION", latest)
dbutils.notebook.exit(str(latest))
