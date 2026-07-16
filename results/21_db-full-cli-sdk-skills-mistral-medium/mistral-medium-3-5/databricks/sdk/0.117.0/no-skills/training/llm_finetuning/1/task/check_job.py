#!/usr/bin/env python3
"""Check the job status and output."""
import os

from databricks.sdk import WorkspaceClient

# Initialize the workspace client
w = WorkspaceClient()

# Environment variables
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]
JOB_NAME = f"{PREFIX}_ftjob65e929"

# Get current user
current_user = w.current_user.me().user_name
print(f"Current user: {current_user}")

# List jobs
jobs = w.jobs.list()
for job in jobs:
    if job.name == JOB_NAME:
        print(f"Found job: {job.job_id}, name: {job.name}")
        # List runs for this job
        runs = list(w.jobs.list_runs(job_id=job.job_id, limit=5))
        for run in runs:
            print(f"  Run {run.run_id}: state={run.state}, result_state={run.result_state}")
            try:
                run_output = w.jobs.get_run_output(run.run_id)
                print(f"  Output: {run_output}")
            except Exception as e:
                print(f"  Error getting output: {e}")
