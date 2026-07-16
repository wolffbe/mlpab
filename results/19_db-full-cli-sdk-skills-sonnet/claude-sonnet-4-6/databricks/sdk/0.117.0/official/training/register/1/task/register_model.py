import os
import json
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

schema = os.environ["MLPAB_DATABRICKS_SCHEMA"]
model_name = "churnmodelcda532"
full_name = f"{schema}.{model_name}"
run_id = "b54c7b21f41745acbc5e85eac633bb8a"

with open("data/metrics.json") as f:
    metrics = json.load(f)

# Finish the run
print("Finishing the run...")
try:
    w.experiments.update_run(run_id=run_id, status="FINISHED")
    print("Run finished")
except Exception as e:
    print(f"Error finishing run: {e}")

# Final verification
print("\n=== Final verification ===")
mv = w.model_versions.get(full_name=full_name, version=1)
print(f"Model name: {model_name}")
print(f"Version: {mv.version}")
print(f"Status: {mv.status}")
print(f"Source: {mv.source}")
print(f"Comment: {mv.comment}")

run = w.experiments.get_run(run_id=run_id)
print(f"\nRun status: {run.run.info.status}")
run_metrics = {m.key: m.value for m in (run.run.data.metrics or [])}
print(f"Run metrics: {run_metrics}")

# Write submission
answers = {
    "model_name": model_name,
    "version": 1,
    "metrics": metrics
}

os.makedirs("submission", exist_ok=True)
with open("submission/answers.json", "w") as f:
    json.dump(answers, f, indent=2)
print(f"\nSubmission written: {answers}")
