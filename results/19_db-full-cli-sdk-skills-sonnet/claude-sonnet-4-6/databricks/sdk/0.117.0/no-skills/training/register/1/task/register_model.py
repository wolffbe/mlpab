import os, json
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.ml import Metric, UpdateRunStatus

w = WorkspaceClient()
me = w.current_user.me()
prefix = os.environ.get('MLPAB_DATABRICKS_PREFIX')
schema = os.environ.get('MLPAB_DATABRICKS_SCHEMA')
catalog_name, schema_name = schema.split('.')
model_name = 'churnmodelcda532'

with open('data/metrics.json') as f:
    metrics = json.load(f)

print(f'Metrics: {metrics}')

# Create experiment
exp_path = f'/Users/{me.user_name}/{prefix}/churnmodel_experiment'
print(f'Creating experiment at: {exp_path}')
try:
    exp = w.experiments.create_experiment(name=exp_path)
    exp_id = exp.experiment_id
    print(f'Created experiment: {exp_id}')
except Exception as e:
    print(f'Experiment exists, getting: {e}')
    exp = w.experiments.get_by_name(experiment_name=exp_path)
    exp_id = exp.experiment.experiment_id
    print(f'Got experiment: {exp_id}')

# Create a run
run_resp = w.experiments.create_run(experiment_id=exp_id, run_name='churnmodel_v1')
run_id = run_resp.run.info.run_id
print(f'Created run: {run_id}')

# Log metrics
metrics_list = [Metric(key=k, value=v, timestamp=0, step=0) for k, v in metrics.items()]
w.experiments.log_batch(run_id=run_id, metrics=metrics_list)
print('Metrics logged')

# Log model artifact
# Finish run
w.experiments.update_run(run_id=run_id, status=UpdateRunStatus.FINISHED)
print('Run finished')

# Get run info for artifact URI
run_info = w.experiments.get_run(run_id=run_id)
artifact_uri = run_info.run.info.artifact_uri
print(f'Artifact URI: {artifact_uri}')

# Create model version in UC registered model using the run
full_model_name = f'{catalog_name}.{schema_name}.{model_name}'
print(f'Creating UC model version for: {full_model_name}')
try:
    mv = w.model_registry.create_model_version(
        name=full_model_name,
        source=artifact_uri,
        run_id=run_id,
        description='Churn model v1 with auc and accuracy metrics'
    )
    version = mv.model_version.version
    print(f'Created UC model version: {version}')
    # Set metrics as tags
    for k, v in metrics.items():
        w.model_registry.set_model_version_tag(
            name=full_model_name,
            version=version,
            key=k,
            value=str(v)
        )
    print('Tags set on UC model version')
except Exception as e:
    print(f'UC model version error: {e}')
    raise

print(f'Done. Model: {full_model_name}, version: {version}')
