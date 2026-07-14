import os
import sys
import datetime
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs, compute as sdk_compute

w = WorkspaceClient()

PREFIX = os.environ['MLPAB_DATABRICKS_PREFIX']
USER = w.current_user.me().user_name
JOB_NAME = f"{PREFIX}_flakyc8344f"
WORKSPACE_DIR = f"/Users/{USER}/{PREFIX}"

print(f"Job name: {JOB_NAME}")
print(f"User: {USER}")
print(f"Workspace dir: {WORKSPACE_DIR}")

# Upload failing_job.py to workspace
script_content = open("data/failing_job.py", "rb").read()

try:
    w.workspace.mkdirs(WORKSPACE_DIR)
    print(f"Created workspace dir: {WORKSPACE_DIR}")
except Exception as e:
    print(f"Workspace dir may already exist: {e}")

script_path = f"{WORKSPACE_DIR}/failing_job.py"
w.workspace.upload(script_path, script_content, overwrite=True)
print(f"Uploaded script to: {script_path}")

# Create job with email notifications on failure (serverless)
ENV_KEY = "default"
job = w.jobs.create(
    name=JOB_NAME,
    environments=[
        jobs.JobEnvironment(
            environment_key=ENV_KEY,
            spec=sdk_compute.Environment(
                client="1",
            ),
        )
    ],
    tasks=[
        jobs.Task(
            task_key="run_failing_job",
            environment_key=ENV_KEY,
            spark_python_task=jobs.SparkPythonTask(
                python_file=script_path,
                source=jobs.Source.WORKSPACE,
            ),
        )
    ],
    email_notifications=jobs.JobEmailNotifications(
        on_failure=[USER],
    ),
    description=f"flakyc8344f failure alert test job — always fails with ERR-137284",
)

job_id = job.job_id
print(f"Created job ID: {job_id}")

# Run the job once
run = w.jobs.run_now(job_id=job_id)
run_id = run.run_id
print(f"Started run ID: {run_id}")

# Wait for it to complete
print("Waiting for run to complete (expected to fail)...")
result = w.jobs.wait_get_run_job_terminated_or_skipped(run_id=run_id, timeout=600)
print(f"Run result state: {result.state.result_state}")
print(f"Run life cycle state: {result.state.life_cycle_state}")

print(f"\nDone! Job '{JOB_NAME}' ran once and failed as expected.")
print(f"Email failure alert configured for: {USER}")
print(f"Alert description: email notification on_failure for job '{JOB_NAME}' (flakyc8344f)")
