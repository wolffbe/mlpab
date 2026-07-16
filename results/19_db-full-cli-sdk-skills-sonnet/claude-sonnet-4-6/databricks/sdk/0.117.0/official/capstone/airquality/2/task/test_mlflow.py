"""Test mlflow setup with auth on serverless cluster."""
import os
import base64
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import SubmitTask, NotebookTask, Source
from databricks.sdk.service.workspace import ImportFormat, Language

w = WorkspaceClient()
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]
SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]
me = w.current_user.me()
user = me.user_name
NOTEBOOK_DIR = f"/Users/{user}/{PREFIX}"

nb = f'''# Databricks notebook source
# Install mlflow with pydantic v1 compatible version
import subprocess, sys, os

PKG_DIR = "/tmp/mlpkgs_airq2"
os.makedirs(PKG_DIR, exist_ok=True)

ret = subprocess.run([sys.executable, "-m", "pip", "install",
                      "--target", PKG_DIR,
                      "mlflow==2.9.2",
                      "pydantic<2"],
                     capture_output=True, text=True)
print("pip returncode:", ret.returncode)
if ret.returncode != 0:
    print("STDERR:", ret.stderr[-800:])
print("STDOUT:", ret.stdout[-300:])

# Add to sys.path front
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

# Clear mlflow modules
for mod in list(sys.modules.keys()):
    if "mlflow" in mod:
        del sys.modules[mod]

try:
    import mlflow
    print("mlflow version:", mlflow.__version__)
except Exception as e:
    print("mlflow import failed:", e)
    dbutils.notebook.exit(f"FAILED mlflow import: {{e}}")

# COMMAND ----------
# Configure Databricks auth for mlflow

import os, configparser

# Get auth from notebook context
try:
    ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    token = ctx.apiToken().get()
    host = ctx.apiUrl().get()
    print(f"Got auth context: host={{host[:30]}}...")
except Exception as e:
    print(f"Failed to get context: {{e}}")
    host = "https://dbc-2a4591fe-28e4.cloud.databricks.com"
    token = ""

# Write ~/.databrickscfg
home = os.path.expanduser("~")
cfg_path = os.path.join(home, ".databrickscfg")
cfg_content = f"[DEFAULT]\\nhost = {{host}}\\ntoken = {{token}}\\n"
with open(cfg_path, "w") as f:
    f.write(cfg_content)
print(f"Wrote {{cfg_path}}")

# Set env vars
os.environ["DATABRICKS_HOST"] = host
os.environ["DATABRICKS_TOKEN"] = token

# COMMAND ----------
# Test mlflow

import mlflow

try:
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

    exp_name = "/Users/{user}/{PREFIX}/test_mlflow_exp"
    try:
        exp_id = mlflow.create_experiment(exp_name)
        print(f"Created experiment: {{exp_id}}")
    except Exception as e:
        print(f"Experiment exists or error: {{e}}")
        exp_id = mlflow.get_experiment_by_name(exp_name)
        if exp_id:
            print(f"Found existing: {{exp_id.experiment_id}}")

    mlflow.set_experiment(exp_name)

    with mlflow.start_run() as run:
        mlflow.log_metric("test_metric", 42.0)
        mlflow.log_param("test_param", "value")
        run_id = run.info.run_id
        print(f"Run ID: {{run_id}}")

    print("MLflow test PASSED")
    dbutils.notebook.exit(f"SUCCESS run_id={{run_id}}")

except Exception as e:
    import traceback
    tb = traceback.format_exc()
    print(f"MLflow test FAILED: {{e}}")
    print(tb)
    dbutils.notebook.exit(f"FAILED: {{e}}")
'''

nb_path = f"{NOTEBOOK_DIR}/test_mlflow"
w.workspace.import_(
    path=nb_path,
    format=ImportFormat.SOURCE,
    language=Language.PYTHON,
    content=base64.b64encode(nb.encode()).decode(),
    overwrite=True,
)

run = w.jobs.submit(
    run_name=f"{PREFIX}_test_mlflow",
    tasks=[SubmitTask(task_key="test", notebook_task=NotebookTask(notebook_path=nb_path, source=Source.WORKSPACE))]
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
                print("OUTPUT:", out.notebook_output.result)
            if out and out.error:
                print("ERROR:", out.error[:1000])
            if out and out.error_trace:
                print("TRACE:", out.error_trace[:1000])
        break
    print(f"  {lc}/{rs}")
    time.sleep(10)
