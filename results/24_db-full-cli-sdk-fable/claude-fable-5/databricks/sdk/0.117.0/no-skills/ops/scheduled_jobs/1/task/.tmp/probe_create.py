from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import (
    Task, NotebookTask, CronSchedule, PauseStatus,
    TriggerSettings, PeriodicTriggerConfiguration,
    PeriodicTriggerConfigurationTimeUnit, JobSettings,
)

w = WorkspaceClient()
user = w.current_user.me().user_name
notebook_path = f"/Users/{user}/mlpab23fe6e/heartbeat"

task = Task(task_key="heartbeat",
            notebook_task=NotebookTask(notebook_path=notebook_path))

# 1. create WITHOUT any schedule
try:
    created = w.jobs.create(name="heartbeat514afe", tasks=[task])
    print("create-no-schedule OK, job_id:", created.job_id)
    job_id = created.job_id
except Exception as e:
    print("create-no-schedule FAILED:", type(e).__name__, e)
    raise SystemExit(1)

# 2. try adding a cron schedule via update
try:
    w.jobs.update(job_id=job_id, new_settings=JobSettings(
        schedule=CronSchedule(quartz_cron_expression="0 0 * * * ?",
                              timezone_id="UTC",
                              pause_status=PauseStatus.UNPAUSED)))
    print("update-add-cron OK")
except Exception as e:
    print("update-add-cron FAILED:", type(e).__name__, e)
    # 3. try a periodic trigger instead
    try:
        w.jobs.update(job_id=job_id, new_settings=JobSettings(
            trigger=TriggerSettings(periodic=PeriodicTriggerConfiguration(
                interval=1, unit=PeriodicTriggerConfigurationTimeUnit.HOURS))))
        print("update-add-periodic-trigger OK")
    except Exception as e2:
        print("update-add-periodic-trigger FAILED:", type(e2).__name__, e2)

job = w.jobs.get(job_id)
print("final settings schedule:", job.settings.schedule)
print("final settings trigger:", job.settings.trigger)
