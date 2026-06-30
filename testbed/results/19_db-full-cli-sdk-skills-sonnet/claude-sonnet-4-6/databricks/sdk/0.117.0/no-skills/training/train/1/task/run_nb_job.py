import os
import sys
from datetime import timedelta
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs

w = WorkspaceClient()
me = w.current_user.me().user_name
prefix = os.environ['MLPAB_DATABRICKS_PREFIX']

nb_name = sys.argv[1] if len(sys.argv) > 1 else 'fe_test_nb'
nb_path = f'/Users/{me}/{prefix}/{nb_name}'
job_suffix = sys.argv[2] if len(sys.argv) > 2 else 'test_job'

job = w.jobs.create(
    name=f'{prefix}_{job_suffix}',
    tasks=[
        jobs.Task(
            task_key='run',
            notebook_task=jobs.NotebookTask(notebook_path=nb_path),
        )
    ]
)
print('Created job:', job.job_id)

run = w.jobs.run_now(job_id=job.job_id)
print('Started run:', run.run_id)

completed = w.jobs.wait_get_run_job_terminated_or_skipped(
    run_id=run.run_id,
    timeout=timedelta(seconds=300)
)
print('Run state:', completed.state.life_cycle_state)
print('Result state:', completed.state.result_state)

for task in (completed.tasks or []):
    if task.run_id:
        output = w.jobs.get_run_output(run_id=task.run_id)
        if output.notebook_output and output.notebook_output.result:
            print('Output:', output.notebook_output.result)
        if output.error:
            print('Error:', output.error[:500])
        if output.error_trace:
            print('Trace:', output.error_trace[:1000])
        if output.logs:
            print('Logs:', output.logs[:500])
