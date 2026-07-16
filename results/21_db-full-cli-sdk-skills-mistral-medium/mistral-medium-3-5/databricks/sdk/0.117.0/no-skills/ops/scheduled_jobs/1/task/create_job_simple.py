#!/usr/bin/env python3
"""Create a recurring Databricks job for heartbeat.py - simpler approach"""

import os
import time
import json
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs, compute

# Environment variables
MLPAB_DATABRICKS_PREFIX = os.environ.get('MLPAB_DATABRICKS_PREFIX', 'mlpabe27f03')

# Job configuration
JOB_NAME = f"{MLPAB_DATABRICKS_PREFIX}_heartbeatd0d7ba"
WORKSPACE_PATH = f"/Users/benedict@logicalclocks.com/{MLPAB_DATABRICKS_PREFIX}/heartbeat.py"

print(f"Creating job: {JOB_NAME}")
print(f"Workspace path: {WORKSPACE_PATH}")

# Initialize the workspace client
w = WorkspaceClient()

# First, upload the heartbeat.py script to the workspace
print("Uploading heartbeat.py to workspace...")
with open('data/heartbeat.py', 'r') as f:
    script_content = f.read()

# Create the directory first
import posixpath
dir_path = posixpath.dirname(WORKSPACE_PATH)
w.workspace.mkdirs(dir_path)
print(f"Directory created: {dir_path}")

# Upload the script
w.workspace.upload(WORKSPACE_PATH, content=script_content, overwrite=True)
print(f"Script uploaded to {WORKSPACE_PATH}")

# Create the job with a recurring schedule using a notebook task
print("Creating recurring job...")

# Create a notebook that runs the Python script
notebook_content = f"# Databricks notebook source\n# MAGIC %python\n{script_content}"
NOTESBOOK_PATH = f"/Users/benedict@logicalclocks.com/{MLPAB_DATABRICKS_PREFIX}/heartbeat_notebook"

w.workspace.upload(NOTESBOOK_PATH, content=notebook_content, overwrite=True)
print(f"Notebook uploaded to {NOTESBOOK_PATH}")

# Define the job settings - use notebook task instead
job = w.jobs.create(
    name=JOB_NAME,
    tasks=[
        jobs.Task(
            task_key="heartbeat_task",
            notebook_task=jobs.NotebookTask(
                notebook_path=NOTESBOOK_PATH
            ),
            existing_cluster_id="1203-211352-8qhq89x8"  # Use an existing cluster if available
        )
    ],
    schedule=jobs.CronSchedule(
        quartz_cron_expression="0 0 0/1 * * ?",  # Hourly schedule in Quartz syntax
        timezone_id="UTC"
    ),
    max_concurrent_runs=1
)

job_id = job.job_id
print(f"Job created with ID: {job_id}")

# Trigger one run manually
print("Triggering first run...")
run = w.jobs.run_now(job_id=job_id)
run_id = run.run_id
print(f"Run triggered with ID: {run_id}")

# Wait for the run to complete
print("Waiting for run to complete...")
max_attempts = 30
attempt = 0

while attempt < max_attempts:
    attempt += 1
    run_info = w.jobs.get_run(run_id=run_id)
    state = run_info.state
    
    if state and state.life_cycle_state:
        print(f"Run state: {state.life_cycle_state}")
        if state.life_cycle_state in ["TERMINATED", "COMPLETED"]:
            print("Run completed successfully!")
            break
        elif state.life_cycle_state == "RUNNING":
            print("Run is still running...")
        else:
            print(f"Run state: {state.life_cycle_state}")
    
    time.sleep(10)

# Create submission directory and write answers.json
print("Writing submission/answers.json...")
os.makedirs('submission', exist_ok=True)

answers = {
    "job_name": "heartbeatd0d7ba"
}

with open('submission/answers.json', 'w') as f:
    json.dump(answers, f, indent=2)

print("Done! Submission file created.")
print(f"Job name: {JOB_NAME}")
print(f"Job ID: {job_id}")
print(f"Run ID: {run_id}")