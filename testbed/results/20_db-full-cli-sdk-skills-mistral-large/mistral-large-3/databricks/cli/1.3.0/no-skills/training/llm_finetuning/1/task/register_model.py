# Databricks notebook source
import mlflow
import json

mlflow.set_registry_uri('databricks-uc')
model_name = 'workspace.mlpab844883.ftmodelcca198'

with open('metrics.json', 'r') as f:
    metrics = json.load(f)

model_uri = 'dbfs:/Volumes/workspace/mlpab844883/artifacts/finetuned_model.npz'
result = mlflow.register_model(model_uri, model_name)

client = mlflow.tracking.MlflowClient()
client.update_model_version(
    name=model_name,
    version=result.version,
    description='Fine-tuned character-level language model',
)

client.update_model_version(
    name=model_name,
    version=result.version,
    tags={
        'eval_loss': str(metrics['eval_loss']),
        'base_eval_loss': str(metrics['base_eval_loss']),
    },
)
