"""Create job flakyc8344f with failure notification, run it once, produce answers.json."""
import json
import os
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import compute
from databricks.sdk.service.jobs import (
    JobEmailNotifications,
    JobEnvironment,
    SparkPythonTask,
    Task,
)
from databricks.sdk.service.workspace import ImportFormat, Language

PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]
JOB_BASE = "flakyc8344f"
JOB_NAME = f"{PREFIX}_{JOB_BASE}"

w = WorkspaceClient()

# Get current user
me = w.current_user.me()
user_email = me.user_name
print(f"Current user: {user_email}")

# Upload the failing script to workspace
script_path = f"/Users/{user_email}/{PREFIX}/failing_job.py"
print(f"Uploading script to: {script_path}")

with open("data/failing_job.py", "rb") as f:
    script_content = f.read()

# Ensure parent folder exists by creating it if needed
try:
    w.workspace.mkdirs(path=f"/Users/{user_email}/{PREFIX}")
except Exception as e:
    print(f"mkdirs: {e}")

w.workspace.upload(
    path=script_path,
    content=script_content,
    format=ImportFormat.AUTO,
    overwrite=True,
)
print("Script uploaded.")

# Create the job with failure email notification
# Email notification on_failure mentions the job name so the alert is identifiable
alert_description = f"on_failure email notification for job {JOB_NAME} ({JOB_BASE})"

job = w.jobs.create(
    name=JOB_NAME,
    environments=[
        JobEnvironment(
            environment_key="default",
            spec=compute.Environment(
                client="1",
            ),
        )
    ],
    tasks=[
        Task(
            task_key="run_failing_job",
            environment_key="default",
            spark_python_task=SparkPythonTask(
                python_file=script_path,
            ),
        )
    ],
    email_notifications=JobEmailNotifications(
        on_failure=[user_email],
        no_alert_for_skipped_runs=True,
    ),
)
print(f"Created job: {job.job_id} — {JOB_NAME}")

# Run the job once (expected to fail)
run = w.jobs.run_now(job_id=job.job_id)
run_id = run.run_id
print(f"Started run: {run_id}")

# Wait for the run to complete
print("Waiting for run to finish...")
while True:
    run_info = w.jobs.get_run(run_id=run_id)
    state = run_info.state
    life_cycle = state.life_cycle_state.value if state.life_cycle_state else "UNKNOWN"
    print(f"  state: {life_cycle}")
    if life_cycle in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        result_state = state.result_state.value if state.result_state else "UNKNOWN"
        print(f"  result: {result_state}")
        break
    time.sleep(10)

# Write answers
os.makedirs("submission", exist_ok=True)
answers = {
    "job_name": JOB_BASE,
    "alert": alert_description,
}
with open("submission/answers.json", "w") as f:
    json.dump(answers, f, indent=2)
print(f"Written: submission/answers.json")
print(json.dumps(answers, indent=2))
