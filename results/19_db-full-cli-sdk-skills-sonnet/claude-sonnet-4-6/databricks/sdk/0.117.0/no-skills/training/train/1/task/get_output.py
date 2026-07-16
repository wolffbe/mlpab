import os
import sys
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
run_id = int(sys.argv[1])

run = w.jobs.get_run(run_id=run_id)
for task in (run.tasks or []):
    print(f'Task: {task.task_key}')
    if task.run_id:
        output = w.jobs.get_run_output(run_id=task.run_id)
        if output.notebook_output:
            print(f'Result: {output.notebook_output.result}')
        if output.error:
            print(f'Error: {output.error[:2000]}')
        if output.error_trace:
            print(f'Trace (first 2000): {output.error_trace[:2000]}')
        if output.logs:
            print(f'Logs: {output.logs[:2000]}')
