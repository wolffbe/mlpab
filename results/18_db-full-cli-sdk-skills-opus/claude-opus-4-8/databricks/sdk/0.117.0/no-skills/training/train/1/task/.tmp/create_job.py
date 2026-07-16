import databricks.sdk as dsdk
from databricks.sdk.service import jobs

w = dsdk.WorkspaceClient()
user = w.current_user.me().user_name
nb_path = f"/Users/{user}/mlpab2138eb/trainjoba834e5_runner"
JOB_NAME = "trainjoba834e5"

# Clean up any pre-existing job with this name
for j in w.jobs.list(name=JOB_NAME):
    print("deleting existing job", j.job_id)
    w.jobs.delete(job_id=j.job_id)

task = jobs.Task(
    task_key="run_training",
    notebook_task=jobs.NotebookTask(notebook_path=nb_path),
)
created = w.jobs.create(name=JOB_NAME, tasks=[task])
print("created job id:", created.job_id)
