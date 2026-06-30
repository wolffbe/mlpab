# Databricks notebook source
# MAGIC %pip install mlflow pandas
# COMMAND ----------
dbutils.library.restartPython()
# COMMAND ----------
import mlflow
import math
import pandas as pd
from mlflow.models import infer_signature

mlflow.set_registry_uri("databricks-uc")

MODEL_NAME = "workspace.mlpab4bb87b.scorer78e1bc"


class ScorerModel(mlflow.pyfunc.PythonModel):
    # model weights (fixed at training time -- do not change)
    A = 2.986326
    B = 2.292444
    C = 1.035017
    D = -1.740277

    def _trigram_weight(self, tri):
        o0, o1, o2 = (ord(ch) for ch in tri)
        return math.sin(self.A * o0 + self.B * o1 + self.C * o2 + self.D)

    def _score(self, text):
        ll = 0.0
        for i in range(len(text) - 2):
            ll += self._trigram_weight(text[i:i + 3])
        return round(ll, 6)

    def predict(self, context, model_input, params=None):
        if isinstance(model_input, pd.DataFrame):
            texts = model_input.iloc[:, 0].tolist()
        elif isinstance(model_input, dict):
            key = list(model_input.keys())[0]
            texts = model_input[key]
        else:
            texts = list(model_input)
        return [{"score": self._score(str(t))} for t in texts]


example = pd.DataFrame({"text": ["hello world"]})
m = ScorerModel()
preds = m.predict(None, example)
print("sanity score:", preds)
sig = infer_signature(example, preds)

with mlflow.start_run() as run:
    info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=ScorerModel(),
        signature=sig,
        input_example=example,
        registered_model_name=MODEL_NAME,
    )

client = mlflow.tracking.MlflowClient()
versions = client.search_model_versions("name='%s'" % MODEL_NAME)
latest = max(int(v.version) for v in versions)
print("REGISTERED_MODEL_VERSION=%d" % latest)
dbutils.notebook.exit(str(latest))
