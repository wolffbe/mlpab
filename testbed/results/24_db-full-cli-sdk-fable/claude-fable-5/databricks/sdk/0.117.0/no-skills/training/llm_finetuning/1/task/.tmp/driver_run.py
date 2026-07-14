"""Run the fine-tune job, wait, then fetch metrics and verify registration."""
import datetime
import json
import os

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
JOB_ID = 433049951332606
VOL_PATH = "/Volumes/workspace/mlpab05c114/ftvol79b056"
MODEL_NAME = "workspace.mlpab05c114.ftmodel79b056"

run = w.jobs.run_now(job_id=JOB_ID).result(timeout=datetime.timedelta(minutes=40))
state = run.status.state if run.status else run.state
print("run finished:", state)

for t in run.tasks or []:
    out = w.jobs.get_run_output(t.run_id)
    if out.error:
        print("TASK ERROR:", out.error)
    if out.logs:
        print("TASK LOGS (tail):", out.logs[-3000:])

# Read metrics.json back through the platform
resp = w.files.download(f"{VOL_PATH}/metrics.json")
metrics = json.loads(resp.contents.read().decode())
print("metrics from platform:", metrics)

# Verify the registered model + version
m = w.registered_models.get(MODEL_NAME)
print("registered model:", m.full_name)
mv = w.model_versions.get(MODEL_NAME, 1)
print("model version 1 status:", mv.status)

os.makedirs("submission", exist_ok=True)
answers = {
    "job_name": "ftjob79b056",
    "model_name": "ftmodel79b056",
    "eval_loss": metrics["eval_loss"],
    "base_eval_loss": metrics["base_eval_loss"],
}
json.dump(answers, open("submission/answers.json", "w"), indent=2)
print("wrote submission/answers.json:", answers)
