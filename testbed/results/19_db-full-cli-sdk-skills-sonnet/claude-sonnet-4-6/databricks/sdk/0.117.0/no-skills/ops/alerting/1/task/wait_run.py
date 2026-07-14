import datetime
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

run_id = 1031262102368591
job_id = 781464702344795

print(f"Waiting for run {run_id}...")
result = w.jobs.wait_get_run_job_terminated_or_skipped(
    run_id=run_id,
    timeout=datetime.timedelta(seconds=600)
)
print(f"Run result state: {result.state.result_state}")
print(f"Run life cycle state: {result.state.life_cycle_state}")
print(f"State message: {result.state.state_message}")
