import os, base64, io
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ImportFormat, Language
from databricks.sdk.service.jobs import Task, NotebookTask

w = WorkspaceClient()
cat, sch = os.environ['MLPAB_DATABRICKS_SCHEMA'].split('.')
prefix = os.environ['MLPAB_DATABRICKS_PREFIX']
user = w.current_user.me().user_name
vol_path = f'/Volumes/{cat}/{sch}/trainvola834e5'

base_dir = f'/Users/{user}/{prefix}'
w.workspace.mkdirs(base_dir)
nb_path = f'{base_dir}/run_trainjoba834e5'

notebook_src = f'''# Databricks notebook source
import os, shutil, tempfile, runpy
vol = "{vol_path}"
work = tempfile.mkdtemp()
for f in ["train.csv", "score.csv", "train_model.py"]:
    shutil.copy(os.path.join(vol, f), os.path.join(work, f))
os.chdir(work)
runpy.run_path("train_model.py", run_name="__main__")
shutil.copy(os.path.join(work, "predictions.csv"), os.path.join(vol, "predictions.csv"))
print("PREDICTIONS WRITTEN to", os.path.join(vol, "predictions.csv"))
'''

w.workspace.upload(nb_path, io.BytesIO(notebook_src.encode()),
                   format=ImportFormat.SOURCE, language=Language.PYTHON, overwrite=True)
print('notebook uploaded:', nb_path)

job_name = f'{prefix}_trainjoba834e5'
# clean up any prior job with same name
for j in w.jobs.list(name=job_name):
    w.jobs.delete(job_id=j.job_id)
    print('deleted prior job', j.job_id)

created = w.jobs.create(
    name=job_name,
    tasks=[Task(task_key='train', notebook_task=NotebookTask(notebook_path=nb_path))],
)
print('JOB_ID', created.job_id)
print('JOB_NAME', job_name)
with open('.tmp/job_id.txt', 'w') as f:
    f.write(str(created.job_id))
