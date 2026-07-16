"""
Script to run the fine-tuning job on Databricks using serverless compute with an environment.
"""
import json
import os
import time
from pathlib import Path

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs, catalog

# Environment variables
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]
SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]  # catalog.schema
CATALOG, SCHEMA_NAME = SCHEMA.split(".")
JOB_NAME = f"{PREFIX}_ftjobcca198"
MODEL_NAME = f"{PREFIX}_ftmodelcca198"
VOLUME_NAME = f"{PREFIX}_volume"

# Local paths
DATA_DIR = Path("data")
SUBMISSION_DIR = Path("submission")
SUBMISSION_DIR.mkdir(exist_ok=True)

# Initialize Databricks client
w = WorkspaceClient()

# Create a volume to store the data
volume_path = f"/Volumes/{CATALOG}/{SCHEMA_NAME}/{VOLUME_NAME}"
try:
    w.volumes.create(
        catalog_name=CATALOG,
        schema_name=SCHEMA_NAME,
        name=VOLUME_NAME,
        volume_type=catalog.VolumeType.MANAGED,
    )
except Exception as e:
    print(f"Volume may already exist: {e}")

# Upload data to the volume
for file in DATA_DIR.glob("*"):
    if file.is_file():
        target_path = f"{volume_path}/{file.name}"
        with file.open("rb") as f:
            w.files.upload(target_path, f, overwrite=True)

# Create the job using serverless compute with an environment
environment = jobs.JobEnvironment(
    environment_key="serverless_env",
    spec=jobs.EnvironmentSpec(
        client=""""
name: serverless-env
dependencies:
  - numpy
  - python==3.10
"""
        )
    ),
)

job = w.jobs.create(
    name=JOB_NAME,
    tasks=[
        jobs.Task(
            task_key="finetune",
            environment_key="serverless_env",
            spark_python_task=jobs.SparkPythonTask(
                python_file=f"dbfs:{volume_path}/finetune_model.py",
            ),
            timeout_seconds=3600,
        )
    ],
    environments=[environment],
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

w.files.download(f"{volume_path}/metrics.json", metrics_path.open("wb"))
w.files.download(f"{volume_path}/finetuned_model.npz", model_path.open("wb"))

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