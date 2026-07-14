import base64
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs, workspace

w = WorkspaceClient()
me = w.current_user.me().user_name
print("user:", me)

base = f"/Users/{me}/mlpabb2baa7"
w.workspace.mkdirs(base)

nb_path = f"{base}/log_scorer_model"
with open(".tmp/log_model_notebook.py", "rb") as f:
    content = base64.b64encode(f.read()).decode()

w.workspace.import_(
    path=nb_path,
    format=workspace.ImportFormat.SOURCE,
    language=workspace.Language.PYTHON,
    content=content,
    overwrite=True,
)
print("uploaded notebook:", nb_path)

run = w.jobs.submit_and_wait(
    run_name="mlpabb2baa7_log_scorer_model",
    tasks=[
        jobs.SubmitTask(
            task_key="log_model",
            notebook_task=jobs.NotebookTask(notebook_path=nb_path),
        )
    ],
    timeout=None,
)
print("run state:", run.state)
task_run_id = run.tasks[0].run_id
out = w.jobs.get_run_output(task_run_id)
print("notebook output:", out.notebook_output)
if out.error:
    print("ERROR:", out.error)
    print(out.error_trace)
