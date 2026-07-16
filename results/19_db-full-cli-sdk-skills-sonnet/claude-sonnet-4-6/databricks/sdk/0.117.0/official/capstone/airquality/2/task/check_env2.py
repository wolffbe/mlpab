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
import subprocess

result = subprocess.run(["pip", "list"], capture_output=True, text=True)
output_lines = []
for line in result.stdout.split("\\n"):
    if any(pkg in line.lower() for pkg in ["mlflow", "sklearn", "scikit", "pydantic", "typing_ext", "numpy", "pandas"]):
        output_lines.append(line)

info = ["Python: " + sys.version[:20]]
info.extend(output_lines)

# COMMAND ----------
dbutils.notebook.exit("\\n".join(info))
'''

nb_path = f"{NOTEBOOK_DIR}/check_env2"
w.workspace.import_(
    path=nb_path,
    format=ImportFormat.SOURCE,
    language=Language.PYTHON,
    content=base64.b64encode(nb.encode()).decode(),
    overwrite=True,
)

run = w.jobs.submit(
    run_name=f"{PREFIX}_check_env2",
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
    if lc in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        for t in (info.tasks or []):
            try:
                out = w.jobs.get_run_output(run_id=t.run_id)
                if out and out.notebook_output and out.notebook_output.result:
                    print("OUTPUT:")
                    print(out.notebook_output.result)
                if out and out.error:
                    print("ERROR:", out.error[:1000])
                if out and out.error_trace:
                    print("TRACE:", out.error_trace[:1000])
            except Exception as ex:
                print(f"No output: {ex}")
        break
    print(f"  {lc}/{rs}")
    time.sleep(10)
