#!/usr/bin/env python3
"""Create job flaky175a29 with failure alert and run it once."""
import os
import json
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import (
    Task,
    NotebookTask,
    Source,
    JobEmailNotifications,
)
from databricks.sdk.service.workspace import ImportFormat

# Environment variables
PREFIX = os.environ.get("MLPAB_DATABRICKS_PREFIX", "mlpabd058b2")

# Job configuration
JOB_NAME = f"{PREFIX}_flaky175a29"
ALERT_NAME = "flaky175a29_failure_alert"

# Get the current user
w = WorkspaceClient()
user = w.current_user.me()
USER_EMAIL = user.emails[0].value  # Get the primary email
USER_PATH = f"/Users/{USER_EMAIL}/{PREFIX}"
NOTEBOOK_PATH = f"{USER_PATH}/failing_job"

print(f"Creating job: {JOB_NAME}")
print(f"User: {USER_EMAIL}")
print(f"Notebook path: {NOTEBOOK_PATH}")

# Create the workspace directory
print("\nCreating workspace directory...")
w.workspace.mkdirs(USER_PATH)
print(f"Created directory: {USER_PATH}")

# Create a notebook from the Python script
print("\nCreating notebook from failing_job.py...")
with open("data/failing_job.py", "r") as f:
    script_content = f.read()

# Create a notebook with the Python code
notebook_content = {
    "cells": [
        {
            "cell_type": "code",
            "language": "python",
            "source": script_content
        }
    ]
}

# Upload the notebook
w.workspace.upload(NOTEBOOK_PATH, content=json.dumps(notebook_content), overwrite=True, format=ImportFormat.JUPYTER)
print(f"Uploaded notebook to {NOTEBOOK_PATH}")

# Create the job with email notifications on failure
# The alert name is mentioned in the job description and in the email address
print("\nCreating job with failure alert...")
job = w.jobs.create(
    name=JOB_NAME,
    description=f"Flaky job for testing failure alerts. Alert: {ALERT_NAME}",
    tasks=[
        Task(
            task_key="run_failing_job",
            notebook_task=NotebookTask(
                notebook_path=NOTEBOOK_PATH,
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
print(f"Job name: {job.settings.name if hasattr(job.settings, 'name') else job.job_id}")

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
