# Databricks notebook source
# MAGIC %pip install mlflow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json
from mlflow.tracking import MlflowClient

MODEL_NAME = "workspace.mlpabbe5553.churnmodel7948d0"
METRICS = {"auc": 0.8524, "accuracy": 0.7254}

client = MlflowClient(registry_uri="databricks-uc")

# Set clean-key tags on the existing version 1 (no new version created)
client.set_model_version_tag(MODEL_NAME, "1", "auc", str(METRICS["auc"]))
client.set_model_version_tag(MODEL_NAME, "1", "accuracy", str(METRICS["accuracy"]))
client.set_model_version_tag(MODEL_NAME, "1", "metrics", json.dumps(METRICS))

mv = client.get_model_version(MODEL_NAME, "1")
print("RUN_ID", mv.run_id)
print("TAGS", mv.tags)

# Confirm metrics are present on the linked run
run = client.get_run(mv.run_id)
print("RUN_METRICS", run.data.metrics)
print("DONE")
