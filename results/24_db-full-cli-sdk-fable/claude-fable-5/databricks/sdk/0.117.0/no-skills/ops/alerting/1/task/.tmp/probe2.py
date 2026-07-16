from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs

w = WorkspaceClient()
me = w.current_user.me().user_name
nb_path = f"/Users/{me}/mlpab752f96/failing_job"

# one-time run submit (no job object)
try:
    r = w.jobs.submit(
        run_name="flaky4f1de7_probe",
        tasks=[jobs.SubmitTask(
            task_key="t",
            notebook_task=jobs.NotebookTask(notebook_path=nb_path),
        )],
    )
    print("submit OK, run_id:", r.run_id)
except Exception as e:
    print("submit FAIL:", str(e)[:140])
