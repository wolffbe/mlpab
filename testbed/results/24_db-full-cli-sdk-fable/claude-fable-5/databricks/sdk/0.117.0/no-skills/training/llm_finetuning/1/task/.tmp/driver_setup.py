"""Set up UC volume, upload inputs, import notebook, create job."""
import base64
import io

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import VolumeType
from databricks.sdk.service.jobs import NotebookTask, Task
from databricks.sdk.service.workspace import ImportFormat, Language

w = WorkspaceClient()
me = w.current_user.me().user_name

CATALOG, SCHEMA = "workspace", "mlpab05c114"
VOL_NAME = "ftvol79b056"
VOL_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOL_NAME}"
NB_DIR = f"/Users/{me}/mlpab05c114"
NB_PATH = f"{NB_DIR}/ftjob79b056_notebook"
JOB_NAME = "mlpab05c114_ftjob79b056"

# 1. Volume
try:
    w.volumes.create(catalog_name=CATALOG, schema_name=SCHEMA, name=VOL_NAME,
                     volume_type=VolumeType.MANAGED)
    print("volume created:", VOL_PATH)
except Exception as e:
    if "already exists" in str(e).lower():
        print("volume exists:", VOL_PATH)
    else:
        raise

# 2. Upload input files
for f in ["base_model.npz", "finetune.txt", "eval.txt", "finetune_model.py"]:
    with open(f"data/{f}", "rb") as fh:
        w.files.upload(f"{VOL_PATH}/{f}", io.BytesIO(fh.read()), overwrite=True)
    print("uploaded", f)

# 3. Notebook
w.workspace.mkdirs(NB_DIR)
src = open(".tmp/ft_notebook.py", "rb").read()
w.workspace.import_(path=NB_PATH, format=ImportFormat.SOURCE,
                    language=Language.PYTHON,
                    content=base64.b64encode(src).decode(), overwrite=True)
print("notebook imported:", NB_PATH)

# 4. Job (serverless: no cluster spec on the notebook task)
existing = [j for j in w.jobs.list(name=JOB_NAME)]
if existing:
    job_id = existing[0].job_id
    print("job exists:", job_id)
else:
    job = w.jobs.create(name=JOB_NAME, tasks=[
        Task(task_key="finetune",
             notebook_task=NotebookTask(notebook_path=NB_PATH)),
    ])
    job_id = job.job_id
    print("job created:", job_id)

print("JOB_ID", job_id)
