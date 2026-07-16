import json

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

for v in w.model_versions.list("workspace.mlpabaf8386.scorer40bb09"):
    mv = w.model_versions.get("workspace.mlpabaf8386.scorer40bb09", v.version)
    print(json.dumps(mv.as_dict(), indent=2, default=str))
