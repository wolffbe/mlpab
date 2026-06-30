import os, datetime
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
job_id = int(open('.tmp/job_id.txt').read().strip())
print('starting run for job', job_id)
run = w.jobs.run_now(job_id=job_id).result(timeout=datetime.timedelta(minutes=30))
print('STATE', run.state.life_cycle_state, run.state.result_state)
print('run_id', run.run_id)
# print task output
for t in run.tasks or []:
    try:
        out = w.jobs.get_run_output(run_id=t.run_id)
        print('--- task', t.task_key, 'output ---')
        if out.notebook_output:
            print(out.notebook_output.result)
        if out.error:
            print('ERROR:', out.error)
    except Exception as e:
        print('output err', e)
