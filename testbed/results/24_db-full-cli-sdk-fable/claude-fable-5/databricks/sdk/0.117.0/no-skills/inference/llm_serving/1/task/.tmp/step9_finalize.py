import json

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

FULL = "workspace.mlpabb2baa7.scorer83d9cf"
resp = w.api_client.do(
    "POST",
    "/api/2.0/mlflow/unity-catalog/model-versions/finalize",
    body={"name": FULL, "version": "1"},
)
print("finalize:", json.dumps(resp, indent=2, default=str))

for e in w.files.list_directory_contents("/Models/workspace/mlpabb2baa7/scorer83d9cf/1"):
    print(e.path, e.file_size)
