import io
import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog, jobs, workspace

w = WorkspaceClient()
me = w.current_user.me().user_name
CAT, SCH = os.environ["MLPAB_DATABRICKS_SCHEMA"].split(".")
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]

# 1) volume + data upload
try:
    w.volumes.create(catalog_name=CAT, schema_name=SCH, name="airqdata",
                     volume_type=catalog.VolumeType.MANAGED)
    print("volume created")
except Exception as e:
    print("volume:", e)

for f in ["airquality_history.csv", "forecast_days.csv"]:
    with open(f"data/{f}", "rb") as fh:
        w.files.upload(f"/Volumes/{CAT}/{SCH}/airqdata/{f}", fh, overwrite=True)
    print("uploaded", f)

# 2) notebook import
nb_dir = f"/Users/{me}/{PREFIX}"
w.workspace.mkdirs(nb_dir)
with open("run/pipeline_notebook.py", "rb") as fh:
    content = fh.read()
nb_path = f"{nb_dir}/airq_pipeline"
w.workspace.upload(nb_path, io.BytesIO(content), format=workspace.ImportFormat.SOURCE,
                   language=workspace.Language.PYTHON, overwrite=True)
print("notebook at", nb_path)

# 3) run as one-time serverless job and wait
run = w.jobs.submit(
    run_name=f"{PREFIX}_airq_pipeline",
    tasks=[jobs.SubmitTask(task_key="pipeline",
                           notebook_task=jobs.NotebookTask(notebook_path=nb_path))],
).result(timeout=None)
print("run state:", run.state.life_cycle_state, run.state.result_state, run.state.state_message)
for t in run.tasks or []:
    try:
        out = w.jobs.get_run_output(t.run_id)
        if out.error:
            print("TASK ERROR:", out.error)
        if out.logs:
            print(out.logs[-6000:])
    except Exception as e:
        print("output fetch:", e)
