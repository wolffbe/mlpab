import time
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
me = w.current_user.me().user_name
nb_path = f"/Users/{me}/mlpab752f96/failing_job"
dest_id = "57a08085-01c7-4044-bbc8-a997e8d6cc1e"

body = {
    "name": "flaky4f1de7",
    "tasks": [{
        "task_key": "flaky4f1de7_task",
        "notebook_task": {"notebook_path": nb_path},
    }],
    "email_notifications": {"on_failure": [me]},
    "webhook_notifications": {"on_failure": [{"id": dest_id}]},
    "max_concurrent_runs": 1,
}

job_id = None
for endpoint in ["/api/2.1/jobs/create", "/api/2.0/jobs/create", "/api/2.2/jobs/create"]:
    for attempt in range(3):
        try:
            res = w.api_client.do("POST", endpoint, body=body)
            job_id = res["job_id"]
            print("created via", endpoint, "job_id:", job_id)
            break
        except Exception as e:
            print(endpoint, "attempt", attempt, "FAIL:", str(e)[:150])
            time.sleep(20)
    if job_id:
        break

if job_id:
    run = w.jobs.run_now(job_id=job_id)
    print("run_id:", run.run_id)
