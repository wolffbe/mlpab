from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs, settings
from databricks.sdk.service.workspace import ImportFormat, Language

w = WorkspaceClient()
me = w.current_user.me().user_name
prefix = "mlpab752f96"
base = f"/Users/{me}/{prefix}"
nb_path = f"{base}/failing_job"

w.workspace.mkdirs(base)
src = open("data/failing_job.py").read()
w.workspace.upload(nb_path, src.encode(), format=ImportFormat.SOURCE,
                   language=Language.PYTHON, overwrite=True)
print("notebook:", nb_path)

dest = w.notification_destinations.create(
    display_name=f"{prefix}_flaky4f1de7_failure_alert",
    config=settings.Config(email=settings.EmailConfig(addresses=[me])),
)
print("destination:", dest.id, dest.display_name)

job = w.jobs.create(
    name=f"{prefix}_flaky4f1de7",
    tasks=[jobs.Task(
        task_key="flaky4f1de7_task",
        notebook_task=jobs.NotebookTask(notebook_path=nb_path),
    )],
    email_notifications=jobs.JobEmailNotifications(on_failure=[me]),
    webhook_notifications=jobs.WebhookNotifications(
        on_failure=[jobs.Webhook(id=dest.id)],
    ),
    max_concurrent_runs=1,
)
print("job_id:", job.job_id)

run = w.jobs.run_now(job_id=job.job_id)
print("run_id:", run.run_id)
