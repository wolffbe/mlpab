import os
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
run_id = 1016250115231750

run = w.jobs.get_run(run_id=run_id)
print('Run state:', run.state.life_cycle_state if run.state else 'None')
print('Tasks:')
for task in (run.tasks or []):
    print(f'  Task: {task.task_key}')
    print(f'  State: {task.state}')
    if task.run_id:
        output = w.jobs.get_run_output(run_id=task.run_id)
        if output.notebook_output:
            print(f'  Notebook output: {output.notebook_output.result}')
        if output.error:
            print(f'  Error: {output.error}')
        if output.error_trace:
            print(f'  Error trace: {output.error_trace[:1000]}')
        if output.logs:
            print(f'  Logs: {output.logs[:1000]}')
