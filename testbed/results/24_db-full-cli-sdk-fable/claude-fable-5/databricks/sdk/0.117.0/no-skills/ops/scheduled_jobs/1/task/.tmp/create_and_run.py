import datetime
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import (
    Task, NotebookTask, CronSchedule, PauseStatus,
)

w = WorkspaceClient()
user = w.current_user.me().user_name
notebook_path = f"/Users/{user}/mlpab23fe6e/heartbeat"
job_name = "heartbeat514afe"

created = None
for attempt in range(8):
    try:
        created = w.jobs.create(
            name=job_name,
            tasks=[
                Task(
                    task_key="heartbeat",
                    notebook_task=NotebookTask(notebook_path=notebook_path),
                )
            ],
            schedule=CronSchedule(
                quartz_cron_expression="0 0 * * * ?",  # hourly, on the hour
                timezone_id="UTC",
                pause_status=PauseStatus.UNPAUSED,
            ),
        )
        break
    except Exception as e:
        print(f"attempt {attempt + 1} failed: {type(e).__name__}: {e}")
        time.sleep(30)

if created is None:
    raise SystemExit("jobs.create never succeeded")

print("job_id:", created.job_id)
job = w.jobs.get(created.job_id)
print("name:", job.settings.name)
print("schedule:", job.settings.schedule)

run = w.jobs.run_now(job_id=created.job_id).result(
    timeout=datetime.timedelta(minutes=30)
)
print("run_id:", run.run_id)
print("state:", run.state)
