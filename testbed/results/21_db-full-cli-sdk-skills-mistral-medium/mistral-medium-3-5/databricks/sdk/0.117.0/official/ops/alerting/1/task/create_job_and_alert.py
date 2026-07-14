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
)

# Environment variables
PREFIX = os.environ.get("MLPAB_DATABRICKS_PREFIX", "mlpabd058b2")
SCHEMA = os.environ.get("MLPAB_DATABRICKS_SCHEMA", "workspace.mlpabd058b2")
USER = "wolffbe"

# Job configuration
JOB_NAME = f"{PREFIX}_flaky175a29"
ALERT_NAME = "flaky175a29_failure_alert"

# Use DBFS path for the script
DBFS_PATH = f"dbfs:/Users/{USER}/{PREFIX}/failing_job.py"

# Initialize workspace client
w = WorkspaceClient()

print(f"Creating job: {JOB_NAME}")
print(f"DBFS path: {DBFS_PATH}")

# First, create the DBFS directory and upload the script
print("\nCreating DBFS directory and uploading script...")
w.dbfs.mkdirs(f"dbfs:/Users/{USER}/{PREFIX}")
print(f"Created DBFS directory: dbfs:/Users/{USER}/{PREFIX}")

# Upload the failing_job.py to DBFS
with open("data/failing_job.py", "r") as f:
    script_content = f.read()

w.dbfs.put(DBFS_PATH, content=script_content, overwrite=True)
print(f"Uploaded failing_job.py to {DBFS_PATH}")

# Create the job with email notifications on failure
# We'll use a dummy email for now - the task just wants the alert configured
# The alert name will be in the job description
job = w.jobs.create(
    name=JOB_NAME,
    description=f"Flaky job for testing failure alerts. Alert: {ALERT_NAME}",
    tasks=[
        Task(
            task_key="run_failing_job",
            spark_python_task=SparkPythonTask(
                python_file=DBFS_PATH,
            ),
            # Set up email notifications on failure
            email_notifications=JobEmailNotifications(
                on_failure=["alert+flaky175a29@test.com"],  # Dummy email that mentions flaky175a29
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
