import io
import json
import os
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog

CATALOG = "workspace"
SCHEMA = "mlpabb2baa7"
MODEL = "scorer83d9cf"
FULL = f"{CATALOG}.{SCHEMA}.{MODEL}"

w = WorkspaceClient()

# 1. volume
try:
    w.volumes.create(
        catalog_name=CATALOG,
        schema_name=SCHEMA,
        name="artifacts",
        volume_type=catalog.VolumeType.MANAGED,
    )
    print("volume created")
except Exception as e:
    print("volume:", e)

# 2. upload model artifacts
base = ".tmp/model"
vol_base = f"/Volumes/{CATALOG}/{SCHEMA}/artifacts/scorer"
for root, _, files in os.walk(base):
    for fn in files:
        local = os.path.join(root, fn)
        rel = os.path.relpath(local, base)
        remote = f"{vol_base}/{rel}"
        with open(local, "rb") as f:
            w.files.upload(remote, io.BytesIO(f.read()), overwrite=True)
        print("uploaded", remote)

# 3. registered model
try:
    w.registered_models.create(catalog_name=CATALOG, schema_name=SCHEMA, name=MODEL)
    print("registered model created")
except Exception as e:
    print("registered model:", e)

# 4. model version from volume source (UC MLflow REST)
resp = w.api_client.do(
    "POST",
    "/api/2.0/mlflow/unity-catalog/model-versions/create",
    body={"name": FULL, "source": vol_base},
)
print("create resp:", json.dumps(resp, indent=2, default=str))
version = resp["model_version"]["version"]
status = resp["model_version"].get("status")

# 5. wait for READY
deadline = time.time() + 600
while status != "READY" and time.time() < deadline:
    time.sleep(10)
    g = w.api_client.do(
        "GET",
        "/api/2.0/mlflow/unity-catalog/model-versions/get",
        query={"name": FULL, "version": str(version)},
    )
    status = g["model_version"].get("status")
    print("status:", status)

print("final:", FULL, "version", version, "status", status)
