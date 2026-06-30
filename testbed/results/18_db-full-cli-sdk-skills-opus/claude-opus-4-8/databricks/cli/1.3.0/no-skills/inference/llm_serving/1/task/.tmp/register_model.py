# Databricks notebook source
# Registers the trigram scorer as an MLflow pyfunc model in Unity Catalog.
import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow.models.signature import infer_signature

CATALOG_SCHEMA = "workspace.mlpab341432"
MODEL_NAME = f"{CATALOG_SCHEMA}.scorer78e1bc"

mlflow.set_registry_uri("databricks-uc")

SCORER_SRC = '''
import math
A = 2.986326
B = 2.292444
C = 1.035017
D = -1.740277

def _trigram_weight(tri):
    o0, o1, o2 = (ord(ch) for ch in tri)
    return math.sin(A * o0 + B * o1 + C * o2 + D)

def score(text):
    ll = 0.0
    for i in range(len(text) - 2):
        ll += _trigram_weight(text[i:i + 3])
    return {"score": round(ll, 6)}
'''


class ScorerModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input):
        import math
        A = 2.986326
        B = 2.292444
        C = 1.035017
        D = -1.740277

        def _trigram_weight(tri):
            o0, o1, o2 = (ord(ch) for ch in tri)
            return math.sin(A * o0 + B * o1 + C * o2 + D)

        def score(text):
            ll = 0.0
            for i in range(len(text) - 2):
                ll += _trigram_weight(text[i:i + 3])
            return round(ll, 6)

        # Normalize input to a list of strings.
        texts = []
        if isinstance(model_input, pd.DataFrame):
            col = model_input.columns[0]
            texts = [str(t) for t in model_input[col].tolist()]
        elif isinstance(model_input, dict):
            # take first value list
            v = list(model_input.values())[0]
            texts = [str(t) for t in v]
        elif isinstance(model_input, (list, tuple)):
            texts = [str(t) for t in model_input]
        else:
            texts = [str(model_input)]
        return [score(t) for t in texts]


example_input = pd.DataFrame({"text": ["hello world"]})
m = ScorerModel()
example_output = m.predict(None, example_input)
signature = infer_signature(example_input, example_output)

with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        artifact_path="scorer",
        python_model=ScorerModel(),
        signature=signature,
        input_example=example_input,
        registered_model_name=MODEL_NAME,
        pip_requirements=["pandas", "mlflow"],
    )

# Find the latest version we just registered.
from mlflow.tracking import MlflowClient
client = MlflowClient()
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
latest = max(int(v.version) for v in versions)
print("REGISTERED_MODEL:", MODEL_NAME)
print("MODEL_VERSION:", latest)
