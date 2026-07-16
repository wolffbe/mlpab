# Databricks notebook source
import mlflow
import mlflow.pyfunc
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec
import numpy as np
import json
import os
import shutil
import tempfile

# COMMAND ----------
VOL_PATH = '/Volumes/workspace/mlpabf4b1c5/mlpabf4b1c5_finetune_vol'
FULL_MODEL_NAME = 'workspace.mlpabf4b1c5.ftmodel0b3133'
USER_PATH = '/Users/benedict@logicalclocks.com/mlpabf4b1c5'

with open(os.path.join(VOL_PATH, 'metrics.json')) as f:
    metrics = json.load(f)
print('Metrics:', metrics)

tmp_dir = tempfile.mkdtemp()
model_src = os.path.join(VOL_PATH, 'finetuned_model.npz')
model_dst = os.path.join(tmp_dir, 'finetuned_model.npz')
shutil.copy2(model_src, model_dst)

# COMMAND ----------
class BigramModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        data = np.load(context.artifacts['model_file'])
        self.logits = data['logits']

    def predict(self, context, model_input):
        return model_input

signature = ModelSignature(
    inputs=Schema([ColSpec("string", "text")]),
    outputs=Schema([ColSpec("string", "result")])
)

# COMMAND ----------
mlflow.set_registry_uri('databricks-uc')
experiment_path = USER_PATH + '/finetune_experiment'
mlflow.set_experiment(experiment_path)

with mlflow.start_run() as run:
    mlflow.log_metric('eval_loss', metrics['eval_loss'])
    mlflow.log_metric('base_eval_loss', metrics['base_eval_loss'])
    mlflow.log_artifact(model_dst, artifact_path='model_files')

    mlflow.pyfunc.log_model(
        artifact_path='model',
        python_model=BigramModelWrapper(),
        artifacts={'model_file': model_dst},
        signature=signature,
        registered_model_name=FULL_MODEL_NAME
    )
    run_id = run.info.run_id
    print(f'Run ID: {run_id}')

print(f'Model registered: {FULL_MODEL_NAME}')

# COMMAND ----------
from mlflow import MlflowClient
client = MlflowClient(registry_uri='databricks-uc')
mv_list = client.search_model_versions(f"name='{FULL_MODEL_NAME}'")
for mv in mv_list:
    print(f'Version {mv.version}: {mv.status}')
    client.set_model_version_tag(FULL_MODEL_NAME, mv.version, 'eval_loss', str(metrics['eval_loss']))
    client.set_model_version_tag(FULL_MODEL_NAME, mv.version, 'base_eval_loss', str(metrics['base_eval_loss']))

print('Done!')
