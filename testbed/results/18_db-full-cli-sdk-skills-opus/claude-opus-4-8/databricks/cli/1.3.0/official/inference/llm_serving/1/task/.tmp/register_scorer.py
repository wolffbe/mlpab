# Databricks notebook source
import mlflow
import math
import pandas as pd
from mlflow.models.signature import infer_signature

REGISTERED_NAME = "workspace.mlpabbe042a.scorer78e1bc_model"

# COMMAND ----------

class ScorerModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params=None):
        import math
        A = 2.986326
        B = 2.292444
        C = 1.035017
        D = -1.740277

        def _tw(tri):
            o0, o1, o2 = (ord(ch) for ch in tri)
            return math.sin(A * o0 + B * o1 + C * o2 + D)

        def _score(text):
            ll = 0.0
            for i in range(len(text) - 2):
                ll += _tw(text[i:i + 3])
            return round(ll, 6)

        if hasattr(model_input, "iloc"):
            col = model_input.columns[0]
            texts = [str(t) for t in model_input[col].tolist()]
        elif isinstance(model_input, dict):
            key = list(model_input.keys())[0]
            texts = [str(t) for t in model_input[key]]
        else:
            texts = [str(t) for t in model_input]
        return [_score(t) for t in texts]

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment("/Users/benedict@logicalclocks.com/mlpabbe042a/scorer_exp")

input_example = pd.DataFrame({"text": ["hello world"]})
signature = infer_signature(input_example, [0.0])

with mlflow.start_run() as run:
    info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=ScorerModel(),
        signature=signature,
        input_example=input_example,
        registered_model_name=REGISTERED_NAME,
    )

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient(registry_uri="databricks-uc")
versions = client.search_model_versions("name='%s'" % REGISTERED_NAME)
latest = max(int(v.version) for v in versions)
print("REGISTERED_VERSION=" + str(latest))
dbutils.notebook.exit(str(latest))
