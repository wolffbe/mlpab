import datetime, os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs

w = WorkspaceClient()
nb_path = f"/Users/{w.current_user.me().user_name}/{os.environ['MLPAB_DATABRICKS_PREFIX']}/run_train178367"

job = w.jobs.create(
    name="trainjob178367",
    tasks=[
        jobs.Task(
            task_key="train",
            notebook_task=jobs.NotebookTask(notebook_path=nb_path),
        )
    ],
)
print("job_id:", job.job_id)

run = w.jobs.run_now_and_wait(job_id=job.job_id, timeout=datetime.timedelta(minutes=30))
print("run state:", run.state.result_state, run.state.state_message)
for t in run.tasks or []:
    out = w.jobs.get_run_output(t.run_id)
    if out.notebook_output:
        print("notebook result:", out.notebook_output.result)
    if out.error:
        print("error:", out.error)
