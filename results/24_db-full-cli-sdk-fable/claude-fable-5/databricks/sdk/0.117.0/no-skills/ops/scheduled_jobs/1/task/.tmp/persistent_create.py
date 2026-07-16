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

deadline = time.monotonic() + 2100  # keep retrying for up to 35 min
created = None
attempt = 0
while time.monotonic() < deadline:
    attempt += 1
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
        msg = str(e).split(". Config:")[0]
        print(f"attempt {attempt}: {type(e).__name__}: {msg}", flush=True)
        time.sleep(60)

if created is None:
    raise SystemExit("jobs.create never succeeded within retry window")

print("job_id:", created.job_id)
job = w.jobs.get(created.job_id)
print("name:", job.settings.name)
print("schedule:", job.settings.schedule)

# trigger one run and wait for it (retry run_now too, in case runs are
# still temporarily disabled right after create starts working)
run = None
run_deadline = time.monotonic() + 600
while time.monotonic() < run_deadline:
    try:
        waiter = w.jobs.run_now(job_id=created.job_id)
        run = waiter.result(timeout=datetime.timedelta(minutes=30))
        break
    except Exception as e:
        msg = str(e).split(". Config:")[0]
        print(f"run_now: {type(e).__name__}: {msg}", flush=True)
        time.sleep(60)

if run is None:
    raise SystemExit("run_now never succeeded / completed")

print("run_id:", run.run_id)
print("result_state:", run.state.result_state)
print("life_cycle_state:", run.state.life_cycle_state)
