"""Check mlflow availability and typing_extensions version on serverless."""
import os
import base64
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import SubmitTask, NotebookTask, Source
from databricks.sdk.service.workspace import ImportFormat, Language

w = WorkspaceClient()
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]
me = w.current_user.me()
user = me.user_name
NOTEBOOK_DIR = f"/Users/{user}/{PREFIX}"

nb = r'''# Databricks notebook source
import subprocess, sys, os

# Check typing_extensions version
result = subprocess.run(["pip", "show", "typing_extensions"], capture_output=True, text=True)
te_info = result.stdout.strip()

# Check if mlflow is importable without install
try:
    import mlflow
    mlflow_v = mlflow.__version__
except ImportError as e:
    mlflow_v = f"NOT FOUND: {e}"

# Check typing_extensions location
import typing_extensions
te_loc = typing_extensions.__file__

info = [
    f"Python: {sys.version[:30]}",
    f"typing_extensions: {te_info[:100]}",
    f"typing_extensions file: {te_loc}",
    f"mlflow: {mlflow_v}",
    f"sys.path[0:3]: {str(sys.path[:3])}",
]

# Check if we can install typing_extensions upgrade
r3 = subprocess.run(["pip", "show", "pydantic"], capture_output=True, text=True)
info.append(f"pydantic: {r3.stdout.strip()[:100]}")

# COMMAND ----------
dbutils.notebook.exit("\n".join(info))
'''

nb_path = f"{NOTEBOOK_DIR}/check_mlflow2"
w.workspace.import_(
    path=nb_path,
    format=ImportFormat.SOURCE,
    language=Language.PYTHON,
    content=base64.b64encode(nb.encode()).decode(),
    overwrite=True,
)

run = w.jobs.submit(
    run_name=f"{PREFIX}_check_mlflow2",
    tasks=[SubmitTask(task_key="check", notebook_task=NotebookTask(notebook_path=nb_path, source=Source.WORKSPACE))]
)
run_id = run.run_id
print(f"Run: {run_id}")

while True:
    info = w.jobs.get_run(run_id=run_id)
    lc = info.state.life_cycle_state.value
    rs = info.state.result_state.value if info.state.result_state else ""
    if lc in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        for t in (info.tasks or []):
            out = w.jobs.get_run_output(run_id=t.run_id)
            if out and out.notebook_output and out.notebook_output.result:
                print("OUTPUT:")
                print(out.notebook_output.result)
            if out and out.error:
                print("ERROR:", out.error[:500])
        break
    print(f"  {lc}/{rs}")
    time.sleep(10)
