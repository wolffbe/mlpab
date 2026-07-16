import datetime
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs

w = WorkspaceClient()
nb_path = "/Users/benedict@hopsworks.ai/mlpab67db84/ccfraud_pipeline"

waiter = None
for attempt in range(30):
    try:
        waiter = w.jobs.submit(
            run_name="mlpab67db84_ccfraud_pipeline",
            tasks=[jobs.SubmitTask(task_key="pipeline", notebook_task=jobs.NotebookTask(notebook_path=nb_path))],
        )
        print("submitted, run_id:", waiter.run_id)
        break
    except Exception as e:
        print(f"attempt {attempt}: {e}", flush=True)
        time.sleep(45)

if waiter is None:
    raise SystemExit("could not submit job after retries")

run = waiter.result(timeout=datetime.timedelta(minutes=40))
print("run state:", run.state)
for t in run.tasks:
    print("task:", t.task_key, t.state)
    out = w.jobs.get_run_output(t.run_id)
    if out.notebook_output and out.notebook_output.result:
        print("OUTPUT:", out.notebook_output.result)
    if out.error:
        print("ERROR:", out.error)
        if out.error_trace:
            print(out.error_trace[:4000])
