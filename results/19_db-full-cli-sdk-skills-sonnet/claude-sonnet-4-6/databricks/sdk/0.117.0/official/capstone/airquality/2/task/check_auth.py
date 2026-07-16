"""Check auth env vars and mlflow setup on serverless."""
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

# Check what env vars are set for auth
env_info = []
for k, v in os.environ.items():
    if any(kw in k.upper() for kw in ["DATABRICKS", "MLFLOW", "TOKEN", "HOST", "SECRET"]):
        # Redact token values
        display = v[:20] + "..." if "TOKEN" in k.upper() or "SECRET" in k.upper() else v
        env_info.append(f"{k}={display}")

# Check home dir
home = os.path.expanduser("~")
env_info.append(f"HOME: {home}")
env_info.append(f"HOME exists: {os.path.exists(home)}")

# Check if databrickscfg exists
cfg_path = os.path.expanduser("~/.databrickscfg")
env_info.append(f".databrickscfg exists: {os.path.exists(cfg_path)}")

# COMMAND ----------
dbutils.notebook.exit("\n".join(sorted(env_info)))
'''

nb_path = f"{NOTEBOOK_DIR}/check_auth"
w.workspace.import_(
    path=nb_path,
    format=ImportFormat.SOURCE,
    language=Language.PYTHON,
    content=base64.b64encode(nb.encode()).decode(),
    overwrite=True,
)

run = w.jobs.submit(
    run_name=f"{PREFIX}_check_auth",
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
                print("ERROR:", out.error[:1000])
            if out and out.error_trace:
                print("TRACE:", out.error_trace[:1000])
        break
    print(f"  {lc}/{rs}")
    time.sleep(10)
