"""Verify registered model + version and write submission/answers.json."""
import json

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
FULL = "workspace.mlpab3d22c1.churnmodel921167"

rm = w.registered_models.get(FULL)
print("registered model:", rm.full_name)

mv = w.model_versions.get(FULL, 1)
print("version:", mv.version, "status:", mv.status)
print("run_id:", mv.run_id)
print("description:", mv.model_version_dependencies is not None, mv.comment)

with open("data/metrics.json") as f:
    metrics = json.load(f)

with open("submission/answers.json", "w") as f:
    json.dump(
        {"model_name": "churnmodel921167", "version": 1, "metrics": metrics}, f, indent=2
    )
print("wrote submission/answers.json")
