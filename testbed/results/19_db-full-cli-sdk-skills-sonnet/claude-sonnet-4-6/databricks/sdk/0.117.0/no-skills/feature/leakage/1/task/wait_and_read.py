import os
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import RunLifeCycleState

w = WorkspaceClient()
schema_full = os.environ['MLPAB_DATABRICKS_SCHEMA']
prefix = os.environ['MLPAB_DATABRICKS_PREFIX']
catalog_name, schema_name = schema_full.split('.')

with open("run_id.txt") as f:
    run_id = int(f.read().strip())

print(f"Waiting for run_id: {run_id}")

# Poll for completion
while True:
    run = w.jobs.get_run(run_id=run_id)
    state = run.state
    lc = state.life_cycle_state if state else None
    result = state.result_state if state else None
    print(f"State: {lc}, Result: {result}")

    if lc in (RunLifeCycleState.TERMINATED, RunLifeCycleState.SKIPPED, RunLifeCycleState.INTERNAL_ERROR):
        print(f"Job finished with result: {result}")
        break

    time.sleep(15)

# Get the output
vol_path = f"/Volumes/{catalog_name}/{schema_name}/{prefix}_data"
try:
    result_bytes = w.files.download(vol_path + "/result.json").contents.read()
    print(f"Result: {result_bytes.decode()}")
    with open("result.json", "wb") as f:
        f.write(result_bytes)
except Exception as e:
    print(f"Error reading result: {e}")
    import traceback
    traceback.print_exc()

# Also get job output/logs
try:
    tasks_output = w.jobs.get_run_output(run_id=run_id)
    print(f"\nNotebook output:")
    if tasks_output.notebook_output:
        print(tasks_output.notebook_output.result)
    if tasks_output.error:
        print(f"Error: {tasks_output.error}")
    if tasks_output.error_trace:
        print(f"Trace: {tasks_output.error_trace}")
except Exception as e:
    print(f"Error getting output: {e}")
