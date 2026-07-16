import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs

w = WorkspaceClient()
me = w.current_user.me().user_name
prefix = "mlpab752f96"
nb_path = f"/Users/{me}/{prefix}/failing_job"
dest_id = "57a08085-01c7-4044-bbc8-a997e8d6cc1e"

job = None
for attempt in range(8):
    try:
        job = w.jobs.create(
            name=f"{prefix}_flaky4f1de7",
            tasks=[jobs.Task(
                task_key="flaky4f1de7_task",
                notebook_task=jobs.NotebookTask(notebook_path=nb_path),
            )],
            email_notifications=jobs.JobEmailNotifications(on_failure=[me]),
            webhook_notifications=jobs.WebhookNotifications(
                on_failure=[jobs.Webhook(id=dest_id)],
            ),
            max_concurrent_runs=1,
        )
        break
    except Exception as e:
        print(f"attempt {attempt}: {e}")
        time.sleep(15)

print("job_id:", job.job_id)
run = w.jobs.run_now(job_id=job.job_id)
print("run_id:", run.run_id)
