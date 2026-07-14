import base64
import json
import os
import time

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

MODEL = "mlpabb2baa7_scorer83d9cf"
DBFS_BASE = "dbfs:/FileStore/mlpabb2baa7/scorer"

base = ".tmp/model"
for root, _, files in os.walk(base):
    for fn in files:
        local = os.path.join(root, fn)
        rel = os.path.relpath(local, base)
        remote = f"{DBFS_BASE}/{rel}"
        with open(local, "rb") as f:
            data = f.read()
        w.dbfs.put(remote, contents=base64.b64encode(data).decode(), overwrite=True)
        print("dbfs put", remote)

print(json.dumps([e.as_dict() for e in w.dbfs.list(DBFS_BASE)], indent=2, default=str))

try:
    w.model_registry.create_model(MODEL)
    print("legacy model created")
except Exception as e:
    print("create_model:", e)

mv = w.model_registry.create_model_version(name=MODEL, source=DBFS_BASE)
print("model version:", mv.model_version.version, mv.model_version.status)

# wait for READY
for _ in range(30):
    got = w.model_registry.get_model_version(MODEL, mv.model_version.version)
    st = got.model_version.status
    print("status:", st)
    if str(st).endswith("READY"):
        break
    time.sleep(5)
