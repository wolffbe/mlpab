from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs
from databricks.sdk.service.workspace import ImportFormat, Language
from datetime import timedelta

w = WorkspaceClient()
user = w.current_user.me().user_name
prefix = "mlpab23ab6a"
base = f"/Users/{user}/{prefix}"
w.workspace.mkdirs(base)
nb = "# Databricks notebook source\nprint('hello serverless', 1+1)\nspark.sql('SELECT 1 as x').show()\n"
nbpath = f"{base}/_probe_nb"
w.workspace.upload(nbpath, nb.encode(), format=ImportFormat.SOURCE,
                   language=Language.PYTHON, overwrite=True)
print("uploaded", nbpath)
run = w.jobs.submit(
    run_name=f"{prefix}_probe",
    tasks=[jobs.SubmitTask(task_key="probe",
        notebook_task=jobs.NotebookTask(notebook_path=nbpath))]
).result(timeout=timedelta(minutes=12))
print("run state:", run.state)
print("tasks:", [(t.task_key, str(t.state)) for t in run.tasks])
