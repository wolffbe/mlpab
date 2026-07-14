"""Check if mlflow is in the Databricks virtualenv."""
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

info = []

# Check virtualenv paths
venv_path = os.environ.get("DATABRICKS_ROOT_VIRTUALENV_ENV", "/databricks/python3")
info.append(f"VENV: {venv_path}")

# Check if mlflow is in the venv
import glob
mlflow_paths = glob.glob(f"{venv_path}/lib/*/site-packages/mlflow*")
info.append(f"mlflow in venv: {mlflow_paths[:3]}")

# Try adding venv to sys.path
venv_site = glob.glob(f"{venv_path}/lib/*/site-packages")
info.append(f"venv site-packages: {venv_site}")

if venv_site:
    for sp in venv_site:
        if sp not in sys.path:
            sys.path.insert(0, sp)
    # Try importing mlflow
    try:
        import mlflow
        info.append(f"mlflow from venv: {mlflow.__version__}")
    except Exception as e:
        info.append(f"mlflow from venv failed: {e}")

# Also check /databricks/spark path
spark_mlflow = glob.glob("/databricks/spark/*/mlflow*")
info.append(f"mlflow in spark: {spark_mlflow[:3]}")

# Check installed pip packages in virtualenv
r = subprocess.run(["pip", "list", "--path", venv_path + "/lib/python3.10/site-packages"],
                   capture_output=True, text=True)
mlflow_line = [l for l in r.stdout.split("\n") if "mlflow" in l.lower()]
info.append(f"mlflow in venv pip: {mlflow_line}")

# COMMAND ----------
dbutils.notebook.exit("\n".join(info))
'''

nb_path = f"{NOTEBOOK_DIR}/check_venv"
w.workspace.import_(
    path=nb_path,
    format=ImportFormat.SOURCE,
    language=Language.PYTHON,
    content=base64.b64encode(nb.encode()).decode(),
    overwrite=True,
)

run = w.jobs.submit(
    run_name=f"{PREFIX}_check_venv",
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
                print("ERROR:", out.error[:2000])
        break
    print(f"  {lc}/{rs}")
    time.sleep(10)
