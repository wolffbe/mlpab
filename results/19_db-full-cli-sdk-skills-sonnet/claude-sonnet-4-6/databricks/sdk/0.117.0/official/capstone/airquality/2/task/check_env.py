"""
Check what's available on Databricks serverless runtime.
"""
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

nb = '''# Databricks notebook source
import sys
print("Python:", sys.version)
print("Sys path:")
for p in sys.path[:10]:
    print(" ", p)

# COMMAND ----------
import subprocess
result = subprocess.run(["pip", "list"], capture_output=True, text=True)
output = result.stdout
# Find relevant packages
for line in output.split("\\n"):
    if any(pkg in line.lower() for pkg in ["mlflow", "sklearn", "scikit", "pydantic", "typing_ext", "numpy", "pandas", "spark"]):
        print(line)
'''

try:
    w.workspace.mkdirs(NOTEBOOK_DIR)
except Exception:
    pass

nb_path = f"{NOTEBOOK_DIR}/check_env"
w.workspace.import_(
    path=nb_path,
    format=ImportFormat.SOURCE,
    language=Language.PYTHON,
    content=base64.b64encode(nb.encode()).decode(),
    overwrite=True,
)

run = w.jobs.submit(
    run_name=f"{PREFIX}_check_env",
    tasks=[
        SubmitTask(
            task_key="check",
            notebook_task=NotebookTask(notebook_path=nb_path, source=Source.WORKSPACE),
        )
    ]
)
run_id = run.run_id
print(f"Run: {run_id}")

while True:
    info = w.jobs.get_run(run_id=run_id)
    lc = info.state.life_cycle_state.value
    rs = info.state.result_state.value if info.state.result_state else ""
    print(f"  {lc}/{rs}")
    if lc in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        for t in (info.tasks or []):
            try:
                out = w.jobs.get_run_output(run_id=t.run_id)
                if out and out.notebook_output and out.notebook_output.result:
                    print("OUTPUT:", out.notebook_output.result[:3000])
                if out and out.error:
                    print("ERROR:", out.error[:500])
            except Exception as ex:
                print(f"No output: {ex}")
        break
    time.sleep(15)
