from databricks.sdk import WorkspaceClient
import sys
w = WorkspaceClient()
parent = int(sys.argv[1])
run = w.jobs.get_run(parent)
for t in run.tasks:
    print('task', t.task_key, t.state.result_state, 'run_id', t.run_id, flush=True)
    try:
        out = w.jobs.get_run_output(t.run_id)
        if out.error:
            print('ERROR:', (out.error or '')[:4000], flush=True)
        if out.error_trace:
            print('TRACE:', (out.error_trace or '')[:4000], flush=True)
        if out.notebook_output and out.notebook_output.result:
            print('RESULT:', out.notebook_output.result[:3000], flush=True)
    except Exception as e:
        print('err', e, flush=True)
