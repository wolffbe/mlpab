import databricks.sdk
from databricks.sdk.service import jobs, compute

w = databricks.sdk.WorkspaceClient()
user = w.current_user.me().user_name
nb_path = '/Users/%s/mlpab695e17/build_scorer_model' % user

task = jobs.SubmitTask(
    task_key='build_model',
    notebook_task=jobs.NotebookTask(notebook_path=nb_path),
    new_cluster=compute.ClusterSpec(
        spark_version='16.4.x-cpu-ml-scala2.12',
        node_type_id='m5d.large',
        num_workers=0,
        spark_conf={'spark.master': 'local[*]', 'spark.databricks.cluster.profile': 'singleNode'},
        custom_tags={'ResourceClass': 'SingleNode'},
        data_security_mode=compute.DataSecurityMode.SINGLE_USER,
    ),
)
print('submitting job run...', flush=True)
run = w.jobs.submit_and_wait(run_name='mlpab695e17_build_scorer', tasks=[task])
print('RUN STATE:', run.state.life_cycle_state, run.state.result_state, flush=True)
print('run_id', run.run_id, flush=True)

# Fetch the task output (notebook exit value = registered version)
out = w.jobs.get_run_output(run.tasks[0].run_id)
print('NOTEBOOK_RESULT:', out.notebook_output.result if out.notebook_output else None, flush=True)
if out.error:
    print('ERROR:', out.error, flush=True)
