"""
Script to run the fine-tuning job on Databricks using a workspace directory and notebook_task.
"""
import json
import os
import time
from pathlib import Path

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs, workspace

# Environment variables
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]
SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]  # catalog.schema
CATALOG, SCHEMA_NAME = SCHEMA.split(".")
JOB_NAME = f"{PREFIX}_ftjobcca198"
MODEL_NAME = f"{PREFIX}_ftmodelcca198"
USER = os.environ.get("DATABRICKS_USER", "unknown")
WORKSPACE_DIR = f"/Users/{USER}/{PREFIX}"

# Local paths
DATA_DIR = Path("data")
SUBMISSION_DIR = Path("submission")
SUBMISSION_DIR.mkdir(exist_ok=True)

# Initialize Databricks client
w = WorkspaceClient()

# Create workspace directory
try:
    w.workspace.mkdirs(WORKSPACE_DIR)
except Exception as e:
    print(f"Workspace directory may already exist: {e}")

# Upload data to the workspace
for file in DATA_DIR.glob("*"):
    if file.is_file():
        target_path = f"{WORKSPACE_DIR}/{file.name}"
        with file.open("rb") as f:
            w.workspace.upload(target_path, f, overwrite=True)

# Create the job using a notebook_task
job = w.jobs.create(
    name=JOB_NAME,
    tasks=[
        jobs.Task(
            task_key="finetune",
            existing_cluster_id="serverless",  # Use serverless compute
            notebook_task=jobs.NotebookTask(
                notebook_path=f"{WORKSPACE_DIR}/finetune_model.py",
            ),
            timeout_seconds=3600,
        )
    ],
    max_concurrent_runs=1,
)

# Run the job and wait for completion
run = w.jobs.run_now(job_id=job.job_id).result()
while True:
    run_status = w.jobs.get_run(run.run_id)
    if run_status.state.life_cycle_state in [jobs.RunLifeCycleState.TERMINATED, jobs.RunLifeCycleState.SKIPPED]:
        break
    if run_status.state.life_cycle_state == jobs.RunLifeCycleState.INTERNAL_ERROR:
        raise RuntimeError(f"Job failed: {run_status.state.state_message}")
    time.sleep(10)

# Download metrics.json and finetuned_model.npz
metrics_path = SUBMISSION_DIR / "metrics.json"
model_path = SUBMISSION_DIR / "finetuned_model.npz"

w.workspace.download(f"{WORKSPACE_DIR}/metrics.json", metrics_path.open("wb"))
w.workspace.download(f"{WORKSPACE_DIR}/finetuned_model.npz", model_path.open("wb"))

# Register the model
with metrics_path.open() as f:
    metrics = json.load(f)

model_version = w.model_versions.create(
    name=f"{SCHEMA}.{MODEL_NAME}",
    source=str(model_path.absolute()),
    run_id=str(run.run_id),
)

# Attach metrics to the model version
w.model_versions.log_metrics(
    name=f"{SCHEMA}.{MODEL_NAME}",
    version=model_version.version,
    metrics={
        "eval_loss": metrics["eval_loss"],
        "base_eval_loss": metrics["base_eval_loss"],
    },
)

# Write submission file
submission = {
    "job_name": JOB_NAME,
    "model_name": f"{SCHEMA}.{MODEL_NAME}",
    "eval_loss": metrics["eval_loss"],
    "base_eval_loss": metrics["base_eval_loss"],
}

with (SUBMISSION_DIR / "answers.json").open("w") as f:
    json.dump(submission, f, indent=2)