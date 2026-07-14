import io
import json
import os

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

FULL = "workspace.mlpabb2baa7.scorer83d9cf"
VERSION = "1"
base = ".tmp/model"
prefix = f"/Models/workspace/mlpabb2baa7/scorer83d9cf/{VERSION}"

for root, _, files in os.walk(base):
    for fn in files:
        local = os.path.join(root, fn)
        rel = os.path.relpath(local, base)
        remote = f"{prefix}/{rel}"
        with open(local, "rb") as f:
            w.files.upload(remote, io.BytesIO(f.read()), overwrite=True)
        print("uploaded", remote)

print("--- listing ---")
for e in w.files.list_directory_contents(prefix):
    print(e.path, e.file_size)

resp = w.api_client.do(
    "POST",
    "/api/2.0/mlflow/unity-catalog/model-versions/finalize",
    body={"name": FULL, "version": VERSION},
)
print("finalize:", json.dumps(resp, indent=2, default=str))
