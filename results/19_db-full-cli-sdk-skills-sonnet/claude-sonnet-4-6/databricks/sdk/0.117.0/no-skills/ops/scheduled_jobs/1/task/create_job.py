"""Create a recurring heartbeat job on Databricks using serverless compute."""
import os
import time
import base64
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import (
    Task, CronSchedule, SparkPythonTask, Source, JobEnvironment
)
from databricks.sdk.service import compute
from databricks.sdk.service.workspace import ImportFormat

w = WorkspaceClient()

prefix = os.environ["MLPAB_DATABRICKS_PREFIX"]
job_name = "heartbeat7bb031"

# Get current user info
me = w.current_user.me()
username = me.user_name
print(f"Current user: {username}")

# Upload the heartbeat script to the workspace
workspace_path = f"/Users/{username}/{prefix}/heartbeat.py"
print(f"Uploading heartbeat.py to {workspace_path}")

with open("data/heartbeat.py", "rb") as f:
    script_content = f.read()

# Create directory if needed
try:
    w.workspace.mkdirs(f"/Users/{username}/{prefix}")
except Exception as e:
    print(f"mkdir note: {e}")

# Import the file
w.workspace.import_(
    path=workspace_path,
    content=base64.b64encode(script_content).decode(),
    overwrite=True,
    format=ImportFormat.AUTO
)
print(f"Uploaded to {workspace_path}")

# Create the recurring job with serverless compute and a schedule
print(f"Creating job: {job_name}")

job = w.jobs.create(
    name=job_name,
    environments=[
        JobEnvironment(
            environment_key="default",
            spec=compute.Environment(
                client="1"
            )
        )
    ],
    tasks=[
        Task(
            task_key="heartbeat",
            environment_key="default",
            spark_python_task=SparkPythonTask(
                python_file=workspace_path,
                source=Source.WORKSPACE
            )
        )
    ],
    schedule=CronSchedule(
        quartz_cron_expression="0 0 * * * ?",  # every hour
        timezone_id="UTC"
    )
)

job_id = job.job_id
print(f"Job created with ID: {job_id}")

# Trigger one immediate run
print("Triggering an immediate run...")
run = w.jobs.run_now(job_id=job_id)
run_id = run.run_id
print(f"Run started with ID: {run_id}")

# Wait for the run to complete
print("Waiting for run to complete...")
for i in range(120):
    run_info = w.jobs.get_run(run_id=run_id)
    state = run_info.state
    life_cycle = state.life_cycle_state if state else None
    result = state.result_state if state else None
    print(f"  [{i*10}s] State: {life_cycle}, Result: {result}")
    if life_cycle and life_cycle.value in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        print(f"Run finished with state: {life_cycle}, result: {result}")
        break
    time.sleep(10)

print(f"\nJob name: {job_name}")
print("Done!")
