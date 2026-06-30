import os
from datetime import timedelta
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs

w = WorkspaceClient()
me = w.current_user.me().user_name
prefix = os.environ['MLPAB_DATABRICKS_PREFIX']
nb_path = f'/Users/{me}/{prefix}/online_enable_nb'

job = w.jobs.create(
    name=f'{prefix}_online_enable_job',
    tasks=[
        jobs.Task(
            task_key='publish',
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

# Get output
for task in (completed.tasks or []):
    print('Task:', task.task_key, 'state:', task.state)
    if task.state and task.state.result_state:
        run_output = w.jobs.get_run_output(run_id=task.run_id)
        if run_output.notebook_output:
            print('Output:', run_output.notebook_output.result)
        if run_output.error:
            print('Error:', run_output.error)
