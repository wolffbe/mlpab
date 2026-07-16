import time

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

MODEL = "mlpabb2baa7_scorer83d9cf"
SRC = "dbfs:/Volumes/workspace/mlpabb2baa7/artifacts/scorer"

try:
    w.model_registry.create_model(MODEL)
    print("legacy model created")
except Exception as e:
    print("create_model:", e)

mv = w.model_registry.create_model_version(name=MODEL, source=SRC)
print("model version:", mv.model_version.version, mv.model_version.status)

for _ in range(30):
    got = w.model_registry.get_model_version(MODEL, mv.model_version.version)
    st = got.model_version.status
    print("status:", st)
    if str(st).endswith("READY"):
        break
    time.sleep(5)
