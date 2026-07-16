import datetime
import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs
from databricks.sdk.service.workspace import ImportFormat, Language

w = WorkspaceClient()
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]
me = w.current_user.me().user_name
nb_dir = f"/Users/{me}/{PREFIX}"
nb_path = f"{nb_dir}/ccfraud_pipeline"

w.workspace.mkdirs(nb_dir)
with open("nb_pipeline.py", "rb") as f:
    w.workspace.upload(nb_path, f, format=ImportFormat.SOURCE, language=Language.PYTHON, overwrite=True)
print("notebook imported:", nb_path)

run = w.jobs.submit(
    run_name=f"{PREFIX}_ccfraud_pipeline",
    tasks=[
        jobs.SubmitTask(
            task_key="pipeline",
            notebook_task=jobs.NotebookTask(notebook_path=nb_path),
        )
    ],
).result(timeout=datetime.timedelta(minutes=45))

print("run state:", run.state)
for t in run.tasks:
    print("task:", t.task_key, t.state)
    out = w.jobs.get_run_output(t.run_id)
    if out.notebook_output and out.notebook_output.result:
        print("OUTPUT:", out.notebook_output.result)
    if out.error:
        print("ERROR:", out.error)
        if out.error_trace:
            print(out.error_trace[:4000])
