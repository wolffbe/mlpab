from databricks.sdk import WorkspaceClient
from databricks.sdk.service import workspace as wsvc
from databricks.sdk.service import jobs
import os, base64, datetime

w = WorkspaceClient()
USER = 'benedict@logicalclocks.com'
PREFIX = os.environ['MLPAB_DATABRICKS_PREFIX']
folder = f'/Users/{USER}/mlpabea3b07'
w.workspace.mkdirs(folder)
nb_path = f'{folder}/fti_pipeline'
with open('.tmp/fti_pipeline.py', 'rb') as f:
    content = f.read()
w.workspace.import_(path=nb_path, format=wsvc.ImportFormat.SOURCE, language=wsvc.Language.PYTHON,
                    content=base64.b64encode(content).decode(), overwrite=True)
print('uploaded notebook', flush=True)

task = jobs.SubmitTask(task_key='fti', notebook_task=jobs.NotebookTask(notebook_path=nb_path))
print('submitting serverless...', flush=True)
wait = w.jobs.submit(run_name=f'{PREFIX}_ccfraud_fti', tasks=[task])
print('run_id', wait.run_id, flush=True)
run = wait.result(timeout=datetime.timedelta(seconds=2000))
print('STATE', run.state.life_cycle_state, run.state.result_state, flush=True)
print('msg', run.state.state_message, flush=True)
try:
    rid = run.tasks[0].run_id
    out = w.jobs.get_run_output(rid)
    print('--- NOTEBOOK OUTPUT ---', flush=True)
    print((out.notebook_output.result or '')[:3000], flush=True)
    if out.error:
        print('--- ERROR ---', flush=True)
        print((out.error or '')[:4000], flush=True)
        print((out.error_trace or '')[:2000], flush=True)
except Exception as e:
    print('output fetch err', e, flush=True)
