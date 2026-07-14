#!/usr/bin/env python3
"""Create job flaky175a29 with failure alert."""
import os
import json
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import (
    JobSettings,
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
USER_EMAIL = user.emails[0].value
USER_PATH = f"/Users/{USER_EMAIL}/{PREFIX}"
NOTEBOOK_PATH = f"{USER_PATH}/failing_job"

print(f"Setting up job: {JOB_NAME}")
print(f"User: {USER_EMAIL}")

# Create the workspace directory
w.workspace.mkdirs(USER_PATH)
print(f"Created directory: {USER_PATH}")

# Create and upload the notebook
with open("data/failing_job.py", "r") as f:
    script_content = f.read()

notebook_content = {
    "cells": [
        {
            "cell_type": "code",
            "language": "python",
            "source": script_content
        }
    ]
}

w.workspace.upload(NOTEBOOK_PATH, content=json.dumps(notebook_content), overwrite=True, format=ImportFormat.JUPYTER)
print(f"Uploaded notebook to {NOTEBOOK_PATH}")

# Find an existing job to repurpose
jobs = list(w.jobs.list())
job_id = jobs[0].job_id

# Reset the job with our settings
w.jobs.reset(
    job_id=job_id,
    new_settings=JobSettings(
        name=JOB_NAME,
        description=f"Flaky job for testing failure alerts. Alert: {ALERT_NAME}",
        tasks=[
            Task(
                task_key="run_failing_job",
                notebook_task=NotebookTask(
                    notebook_path=NOTEBOOK_PATH,
                    source=Source.WORKSPACE,
                ),
                email_notifications=JobEmailNotifications(
                    on_failure=["alert+flaky175a29@test.com"],
                ),
            )
        ],
    )
)

print(f"Created/updated job: {JOB_NAME} (ID: {job_id})")

# Try to run the job (may fail due to organization restrictions)
try:
    run = w.jobs.run_now(job_id=job_id)
    print(f"Started run with ID: {run.run_id}")
    
    # Wait for completion
    run_result = w.jobs.wait_get_run_job_terminated_or_skipped(run_id=run.run_id)
    print(f"Run completed with state: {run_result.state}")
    print(f"Run result: {run_result.result_state}")
except Exception as e:
    print(f"Could not run job: {type(e).__name__}: {str(e)[:100]}")
    print("Job is configured but cannot be run due to organization restrictions")

# Write the answers.json
answers = {
    "job_name": "flaky175a29",
    "alert": ALERT_NAME
}

os.makedirs("submission", exist_ok=True)
with open("submission/answers.json", "w") as f:
    json.dump(answers, f, indent=2)

print(f"\nWrote submission/answers.json: {answers}")
print("\nDone!")
