import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import Task, NotebookTask, Source

w = WorkspaceClient()
schema_full = os.environ['MLPAB_DATABRICKS_SCHEMA']
prefix = os.environ['MLPAB_DATABRICKS_PREFIX']
catalog_name, schema_name = schema_full.split('.')
me = w.current_user.me()
user = me.user_name

notebook_path = f"/Users/{user}/{prefix}/leakage_analysis"
job_name = f"{prefix}_leakage_analysis"

# Use serverless compute (no cluster spec)
run = w.jobs.submit(
    run_name=job_name,
    tasks=[Task(
        task_key="analyze",
        notebook_task=NotebookTask(
            notebook_path=notebook_path,
            source=Source.WORKSPACE
        )
    )]
)
run_id = run.run_id
print(f"Job submitted, run_id: {run_id}")
with open("run_id.txt", "w") as f:
    f.write(str(run_id))
