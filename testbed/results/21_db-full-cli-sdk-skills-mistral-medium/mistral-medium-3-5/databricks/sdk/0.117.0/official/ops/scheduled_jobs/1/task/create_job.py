#!/usr/bin/env python3
"""Create a recurring Databricks job that runs heartbeat.py periodically."""

import os
import time
import json
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs, compute

# Environment variables
DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST")
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN")
MLPAB_DATABRICKS_PREFIX = os.environ.get("MLPAB_DATABRICKS_PREFIX", "mlpabeacd6e")

# Job configuration
JOB_NAME = f"{MLPAB_DATABRICKS_PREFIX}_heartbeatd0d7ba"
WORKSPACE_PATH = f"/Shared/{MLPAB_DATABRICKS_PREFIX}/heartbeat.py"

print(f"Creating recurring job: {JOB_NAME}")
print(f"Workspace path: {WORKSPACE_PATH}")

# Initialize the workspace client
w = WorkspaceClient()

# First, create the parent directory in the workspace
import os
workspace_dir = os.path.dirname(WORKSPACE_PATH)
print(f"Creating workspace directory: {workspace_dir}")
w.workspace.mkdirs(workspace_dir)

# Upload the heartbeat.py script to the workspace
print("Uploading heartbeat.py to workspace...")
with open("data/heartbeat.py", "r") as f:
    script_content = f.read()

# Write the script to the workspace
w.workspace.upload(WORKSPACE_PATH, script_content.encode('utf-8'), overwrite=True)
print(f"Script uploaded to {WORKSPACE_PATH}")

# Create the job with a recurring schedule
# Using a daily schedule (every 24 hours) as a reasonable recurring interval
# Quartz cron syntax: seconds, minutes, hours, day-of-month, month, day-of-week, year(optional)
schedule = jobs.CronSchedule(
    quartz_cron_expression="0 0 0 * * ?",  # Daily at midnight UTC (Quartz syntax)
    timezone_id="UTC",
    pause_status=jobs.PauseStatus.UNPAUSED
)

# Define the environment for serverless compute
# Use the default workspace base environment (AI Latest)
environment = jobs.JobEnvironment(
    environment_key="heartbeat_env",
    spec=compute.Environment(
        base_environment="workspace-base-environments/databricks_ml",
    ),
)

# Define the task to run the Python script
task = jobs.Task(
    task_key="heartbeat_task",
    description="Heartbeat task that prints a line and exits",
    environment_key="heartbeat_env",
    spark_python_task=jobs.SparkPythonTask(
        python_file=WORKSPACE_PATH
    ),
)

# Create the job
job = w.jobs.create(
    name=JOB_NAME,
    description="Recurring heartbeat job",
    schedule=schedule,
    environments=[environment],
    tasks=[task],
    max_concurrent_runs=1,
)

print(f"Job created successfully: {job.job_id}")

# Trigger one run manually
print("Triggering first run...")
run = w.jobs.run_now(job.job_id)
print(f"Run triggered: {run.run_id}")

# Wait for the run to complete
print("Waiting for run to complete...")
max_attempts = 30
for i in range(max_attempts):
    run_info = w.jobs.get_run(run.run_id)
    state = run_info.state
    print(f"Run {run.run_id} state: {state.life_cycle_state} ({state.result_state})")
    
    if state.life_cycle_state in ["TERMINATED", "INTERNAL_ERROR"]:
        print(f"Run completed with result: {state.result_state}")
        break
    
    time.sleep(10)
else:
    print(f"Run did not complete within {max_attempts * 10} seconds")

# Write the submission file
print("Writing submission/answers.json...")
with open("submission/answers.json", "w") as f:
    json.dump({"job_name": "heartbeatd0d7ba"}, f)

print("Done!")
print(f"Job name: heartbeatd0d7ba")
print(f"Job ID: {job.job_id}")
