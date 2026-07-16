import base64
import os

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ImportFormat, Language

w = WorkspaceClient()
me = w.current_user.me().user_name
print("user:", me)

PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]
SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]
base_dir = f"/Users/{me}/{PREFIX}"
w.workspace.mkdirs(base_dir)

notebook = f'''# Databricks notebook source
import math

import mlflow
import pandas as pd
from mlflow.models import infer_signature

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment("{base_dir}/scorer_experiment")


class TrigramScorer(mlflow.pyfunc.PythonModel):
    A = 2.449119
    B = 1.093524
    C = 1.192579
    D = 1.949479

    def _score(self, text):
        ll = 0.0
        for i in range(len(text) - 2):
            o0, o1, o2 = (ord(ch) for ch in text[i:i + 3])
            ll += math.sin(self.A * o0 + self.B * o1 + self.C * o2 + self.D)
        return round(ll, 6)

    def predict(self, context, model_input, params=None):
        texts = model_input["text"].tolist()
        return [self._score(t) for t in texts]


model = TrigramScorer()
sample_in = pd.DataFrame({{"text": ["hello world"]}})
sig = infer_signature(sample_in, model.predict(None, sample_in))

with mlflow.start_run():
    mlflow.pyfunc.log_model(
        "model",
        python_model=model,
        signature=sig,
        input_example=sample_in,
        registered_model_name="{SCHEMA}.scorer83d9cf",
        pip_requirements=["pandas"],
    )
print("registered")
'''

nb_path = f"{base_dir}/register_scorer"
w.workspace.import_(
    path=nb_path,
    format=ImportFormat.SOURCE,
    language=Language.PYTHON,
    content=base64.b64encode(notebook.encode()).decode(),
    overwrite=True,
)
print("notebook:", nb_path)
