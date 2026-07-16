import io, os, time, json
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ImportFormat, Language
from databricks.sdk.service.jobs import SubmitTask, NotebookTask

w = WorkspaceClient()
prefix = os.environ["MLPAB_DATABRICKS_PREFIX"]
me = w.current_user.me().user_name
nb_dir = f"/Users/{me}/{prefix}"
nb_path = f"{nb_dir}/prediction_shift_analysis"

w.workspace.mkdirs(nb_dir)
with open(".tmp/notebook.py", "rb") as f:
    content = f.read()
w.workspace.upload(nb_path, io.BytesIO(content), format=ImportFormat.SOURCE,
                   language=Language.PYTHON, overwrite=True)
print("notebook uploaded:", nb_path)

run = w.jobs.submit(
    run_name=f"{prefix}_prediction_shift_analysis",
    tasks=[SubmitTask(task_key="analyze",
                      notebook_task=NotebookTask(notebook_path=nb_path))],
).result(timeout=__import__("datetime").timedelta(minutes=25))

print("run state:", run.state)
out = w.jobs.get_run_output(run.tasks[0].run_id)
print("notebook output:")
print(out.notebook_output.result if out.notebook_output else out.error)
