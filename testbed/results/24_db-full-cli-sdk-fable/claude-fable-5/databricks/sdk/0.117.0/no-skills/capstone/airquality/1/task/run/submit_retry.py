import os
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs

w = WorkspaceClient()
me = w.current_user.me().user_name
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]
nb_path = f"/Users/{me}/{PREFIX}/airq_pipeline"

waiter = None
for attempt in range(12):
    try:
        waiter = w.jobs.submit(
            run_name=f"{PREFIX}_airq_pipeline",
            tasks=[jobs.SubmitTask(task_key="pipeline",
                                   notebook_task=jobs.NotebookTask(notebook_path=nb_path))],
        )
        print("submitted run_id:", waiter.run_id, flush=True)
        break
    except Exception as e:
        print(f"attempt {attempt}: {e}", flush=True)
        time.sleep(30)

if waiter is None:
    raise SystemExit("job submission permanently rejected")

run = waiter.result(timeout=None)
print("run state:", run.state.life_cycle_state, run.state.result_state, run.state.state_message, flush=True)
for t in run.tasks or []:
    try:
        out = w.jobs.get_run_output(t.run_id)
        if out.error:
            print("TASK ERROR:", out.error, flush=True)
        if out.logs:
            print(out.logs[-8000:], flush=True)
    except Exception as e:
        print("output fetch:", e, flush=True)
