import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs

w = WorkspaceClient()
me = w.current_user.me().user_name
nb_path = f"/Users/{me}/mlpab752f96/failing_job"
dest_id = "57a08085-01c7-4044-bbc8-a997e8d6cc1e"

deadline = time.time() + 1500  # ~25 min of retries
job_id = None
while time.time() < deadline and job_id is None:
    try:
        job = w.jobs.create(
            name="flaky4f1de7",
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
        job_id = job.job_id
        print("JOB CREATED:", job_id, flush=True)
    except Exception as e:
        print("create retry:", str(e)[:90], flush=True)
        time.sleep(30)

if job_id is None:
    print("GIVING UP: job creation still blocked")
    raise SystemExit(1)

# trigger a run (also retried — same throttle may apply)
run_id = None
while time.time() < deadline + 600 and run_id is None:
    try:
        r = w.jobs.run_now(job_id=job_id)
        run_id = r.run_id
        print("RUN TRIGGERED:", run_id, flush=True)
    except Exception as e:
        print("run_now retry:", str(e)[:90], flush=True)
        time.sleep(30)

if run_id is None:
    print("GIVING UP: run trigger still blocked; job exists:", job_id)
    raise SystemExit(2)

# wait for terminal state
while True:
    run = w.jobs.get_run(run_id=run_id)
    st = run.status
    state = st.state.value if st and st.state else None
    print("run state:", state, flush=True)
    if state in ("TERMINATED",):
        term = st.termination_details
        print("termination:", term.code.value if term and term.code else None,
              "-", (term.message or "")[:200] if term else "")
        break
    time.sleep(20)
