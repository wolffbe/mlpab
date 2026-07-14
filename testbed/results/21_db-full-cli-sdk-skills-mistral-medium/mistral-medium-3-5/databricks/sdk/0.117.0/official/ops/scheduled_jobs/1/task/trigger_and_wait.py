#!/usr/bin/env python3
"""Trigger a run of the heartbeat job and wait for it to complete."""

import os
import time
import json
from databricks.sdk import WorkspaceClient

# Environment variables
MLPAB_DATABRICKS_PREFIX = os.environ.get("MLPAB_DATABRICKS_PREFIX", "mlpabeacd6e")
JOB_NAME = f"{MLPAB_DATABRICKS_PREFIX}_heartbeatd0d7ba"

print(f"Finding job: {JOB_NAME}")

# Initialize the workspace client
w = WorkspaceClient()

# Find the job by name
jobs_list = list(w.jobs.list(name=JOB_NAME))
print(f"Found {len(jobs_list)} jobs with name {JOB_NAME}")

if not jobs_list:
    print("No job found, exiting")
    exit(1)

job = jobs_list[0]
print(f"Job ID: {job.job_id}")

# Trigger one run manually
print("Triggering first run...")
run = w.jobs.run_now(job.job_id)
print(f"Run triggered: {run.run_id}")

# Wait for the run to complete
print("Waiting for run to complete...")
max_attempts = 60  # 10 minutes
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

print("Done!")
