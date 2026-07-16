"""Create a recurring heartbeat job on Databricks using a notebook task."""
import os
import base64
import json
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import (
    Task,
    NotebookTask,
    CronSchedule,
    JobEnvironment,
    Source,
)
from databricks.sdk.service import compute
from databricks.sdk.service.workspace import ImportFormat, Language

w = WorkspaceClient()

prefix = os.environ["MLPAB_DATABRICKS_PREFIX"]
me = w.current_user.me()
username = me.user_name
print(f"User: {username}, Prefix: {prefix}")

workspace_dir = f"/Users/{username}/{prefix}"
notebook_path = f"{workspace_dir}/heartbeat"

# Create notebook content (Databricks notebook format)
notebook_content = """\
# Databricks notebook source
\"\"\"Heartbeat — a trivial periodic task. Prints one line and exits 0.\"\"\"
import datetime

TOKEN = "HB-48862277"

print(f"heartbeat {TOKEN} alive at "
      f"{datetime.datetime.now(datetime.timezone.utc).isoformat()}")
"""

try:
    w.workspace.mkdirs(workspace_dir)
    print(f"Created directory: {workspace_dir}")
except Exception as e:
    print(f"Directory note: {e}")

w.workspace.import_(
    path=notebook_path,
    format=ImportFormat.SOURCE,
    language=Language.PYTHON,
    content=base64.b64encode(notebook_content.encode()).decode(),
    overwrite=True,
)
print(f"Uploaded notebook to: {notebook_path}")

# Create the job with serverless compute and hourly schedule
job_name = "heartbeat7bb031"

job = w.jobs.create(
    name=job_name,
    tasks=[
        Task(
            task_key="heartbeat_task",
            notebook_task=NotebookTask(
                notebook_path=notebook_path,
                source=Source.WORKSPACE,
            ),
        )
    ],
    schedule=CronSchedule(
        quartz_cron_expression="0 0 * * * ?",
        timezone_id="UTC",
    ),
)
print(f"Created job: {job_name}, ID: {job.job_id}")

# Trigger one run immediately
run = w.jobs.run_now(job_id=job.job_id)
run_id = run.run_id
print(f"Triggered run ID: {run_id}")

# Wait for completion
while True:
    run_info = w.jobs.get_run(run_id=run_id)
    state = run_info.state
    lc = state.life_cycle_state.value if state.life_cycle_state else "UNKNOWN"
    rs = state.result_state.value if state.result_state else "PENDING"
    print(f"Run state: {lc} / {rs}")
    if lc in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        break
    time.sleep(10)

print(f"Final state: {lc} / {rs}")
print(f"Job name: {job_name}")
print(f"Job ID: {job.job_id}")
