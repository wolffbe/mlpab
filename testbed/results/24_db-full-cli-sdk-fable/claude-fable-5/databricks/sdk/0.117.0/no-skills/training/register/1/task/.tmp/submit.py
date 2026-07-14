"""Local driver: submit serverless one-time job run and wait for completion."""
import datetime

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs
from databricks.sdk.service.compute import Environment

w = WorkspaceClient()

waiter = w.jobs.submit(
    run_name="mlpab3d22c1_register_churnmodel",
    tasks=[
        jobs.SubmitTask(
            task_key="register",
            spark_python_task=jobs.SparkPythonTask(
                python_file="/Workspace/Users/benedict@hopsworks.ai/mlpab3d22c1/register_job.py"
            ),
            environment_key="default",
        )
    ],
    environments=[
        jobs.JobEnvironment(
            environment_key="default",
            spec=Environment(environment_version="3"),
        )
    ],
)
print("submitted run_id", waiter.run_id)
run = waiter.result(timeout=datetime.timedelta(minutes=30))
print("state:", run.status)
for t in run.tasks or []:
    out = w.jobs.get_run_output(t.run_id)
    print("--- task", t.task_key, t.status)
    if out.logs:
        print(out.logs[-4000:])
    if out.error:
        print("ERROR:", out.error)
        if out.error_trace:
            print(out.error_trace[-4000:])
