# Databricks notebook source
import math
import mlflow
import pandas as pd
from mlflow.models import infer_signature

mlflow.set_registry_uri("databricks-uc")


class TrigramScorer(mlflow.pyfunc.PythonModel):
    def _score(self, text):
        ll = 0.0
        for i in range(len(text) - 2):
            o0, o1, o2 = (ord(ch) for ch in text[i:i + 3])
            ll += math.sin(2.449119 * o0 + 1.093524 * o1 + 1.192579 * o2 + 1.949479)
        return round(ll, 6)

    def predict(self, context, model_input):
        if isinstance(model_input, pd.DataFrame):
            texts = model_input["text"].tolist()
        else:
            texts = list(model_input)
        return [self._score(t) for t in texts]


example_in = pd.DataFrame({"text": ["hello world"]})
model = TrigramScorer()
example_out = model.predict(None, example_in)
signature = infer_signature(example_in, example_out)

with mlflow.start_run():
    mlflow.pyfunc.log_model(
        artifact_path="scorer",
        python_model=model,
        signature=signature,
        input_example=example_in,
        registered_model_name="workspace.mlpabb2baa7.scorer_model",
        pip_requirements=["mlflow", "pandas"],
    )

print(model._score("hello world"))
