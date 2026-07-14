"""
Upload notebook and submit as Databricks job.
All ML work runs on the platform.
"""
import os
import base64
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import SubmitTask, NotebookTask, Source
from databricks.sdk.service.workspace import ImportFormat, Language

w = WorkspaceClient()

SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]
CATALOG = SCHEMA.split(".")[0]
DB = SCHEMA.split(".")[1]

me = w.current_user.me()
user = me.user_name
print(f"User: {user}, Schema: {SCHEMA}")

NOTEBOOK_DIR = f"/Users/{user}/{PREFIX}"

# Read data
with open("data/airquality_history.csv") as f:
    history_csv = f.read()
with open("data/forecast_days.csv") as f:
    forecast_csv = f.read()

# Read template and substitute
with open("nb_template.txt") as f:
    nb = f.read()

nb = nb.replace("PLACEHOLDER_SCHEMA", SCHEMA)
nb = nb.replace("PLACEHOLDER_CATALOG", CATALOG)
nb = nb.replace("PLACEHOLDER_DB", DB)
nb = nb.replace("PLACEHOLDER_USER", user)
nb = nb.replace("PLACEHOLDER_PREFIX", PREFIX)
nb = nb.replace("PLACEHOLDER_HISTORY_CSV", history_csv)
nb = nb.replace("PLACEHOLDER_FORECAST_CSV", forecast_csv)

# Create workspace dir
try:
    w.workspace.mkdirs(NOTEBOOK_DIR)
except Exception:
    pass

# Upload
nb_path = f"{NOTEBOOK_DIR}/airquality_pipeline"
w.workspace.import_(
    path=nb_path,
    format=ImportFormat.SOURCE,
    language=Language.PYTHON,
    content=base64.b64encode(nb.encode()).decode(),
    overwrite=True,
)
print(f"Notebook uploaded: {nb_path}")

# Submit - serverless compute is the default in this workspace
run = w.jobs.submit(
    run_name=f"{PREFIX}_airq_pipeline",
    tasks=[
        SubmitTask(
            task_key="pipeline",
            notebook_task=NotebookTask(
                notebook_path=nb_path,
                source=Source.WORKSPACE,
            ),
        )
    ]
)
run_id = run.run_id
print(f"Run ID: {run_id}")

# Wait
while True:
    info = w.jobs.get_run(run_id=run_id)
    state = info.state
    lc = state.life_cycle_state.value if state and state.life_cycle_state else "UNKNOWN"
    rs = state.result_state.value if state and state.result_state else ""
    print(f"  {lc} / {rs}")
    if lc in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        if rs != "SUCCESS":
            try:
                for t in (info.tasks or []):
                    out = w.jobs.get_run_output(run_id=t.run_id)
                    if out and out.error:
                        print(f"Error: {out.error}")
                    if out and out.notebook_output and out.notebook_output.result:
                        print(f"Output: {out.notebook_output.result[:2000]}")
            except Exception as ex:
                print(f"Could not get output: {ex}")
        break
    time.sleep(20)

print(f"Finished: {lc} / {rs}")
