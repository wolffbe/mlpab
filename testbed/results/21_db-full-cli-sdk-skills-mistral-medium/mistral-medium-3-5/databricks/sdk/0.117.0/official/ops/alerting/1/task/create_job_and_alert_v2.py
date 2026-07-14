#!/usr/bin/env python3
"""Create job flaky175a29 with failure alert and run it once."""
import os
import json
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import (
    Task,
    SparkPythonTask,
    Source,
    JobEmailNotifications,
    JobCluster,
    JobEnvironment,
)
from databricks.sdk.service import compute

# Environment variables
PREFIX = os.environ.get("MLPAB_DATABRICKS_PREFIX", "mlpabd058b2")

# Job configuration
JOB_NAME = f"{PREFIX}_flaky175a29"
ALERT_NAME = "flaky175a29_failure_alert"
WORKSPACE_DIR = f"/{PREFIX}_flaky175a29"
SCRIPT_PATH = f"{WORKSPACE_DIR}/failing_job.py"

# Initialize workspace client
w = WorkspaceClient()

print(f"Creating job: {JOB_NAME}")
print(f"Script path: {SCRIPT_PATH}")

# Create the workspace directory and upload the script
print("\nCreating workspace directory and uploading script...")
w.workspace.mkdirs(WORKSPACE_DIR)
print(f"Created directory: {WORKSPACE_DIR}")

with open("data/failing_job.py", "r") as f:
    script_content = f.read()

w.workspace.upload(SCRIPT_PATH, content=script_content, overwrite=True)
print(f"Uploaded failing_job.py to {SCRIPT_PATH}")

# Create the job with email notifications on failure
# The alert name is mentioned in the job description and in the email address
job = w.jobs.create(
    name=JOB_NAME,
    description=f"Flaky job for testing failure alerts. Alert: {ALERT_NAME}",
    tasks=[
        Task(
            task_key="run_failing_job",
            spark_python_task=SparkPythonTask(
                python_file=SCRIPT_PATH,
                source=Source.WORKSPACE,
            ),
            # Set up email notifications on failure
            # The email address mentions flaky175a29 to make the alert identifiable
            email_notifications=JobEmailNotifications(
                on_failure=["alert+flaky175a29@test.com"],
            ),
        )
    ],
)

print(f"\nCreated job with ID: {job.job_id}")
print(f"Job name: {job.name}")

# Now run the job once
print("\nRunning job once (it will fail)...")
run = w.jobs.run_now(job_id=job.job_id)
print(f"Started run with ID: {run.run_id}")

# Wait for the run to complete
print("Waiting for run to complete...")
run_result = w.jobs.wait_get_run_job_terminated_or_skipped(run_id=run.run_id)
print(f"Run completed with state: {run_result.state}")
print(f"Run result: {run_result.result_state}")

# Now write the answers.json
answers = {
    "job_name": "flaky175a29",
    "alert": ALERT_NAME
}

# Create submission directory if it doesn't exist
os.makedirs("submission", exist_ok=True)

with open("submission/answers.json", "w") as f:
    json.dump(answers, f, indent=2)

print(f"\nWrote answers.json: {answers}")
print("\nDone!")
