# Databricks notebook source

# COMMAND ----------
import json, mlflow

mlflow.set_registry_uri("databricks-uc")

client = mlflow.tracking.MlflowClient()

model_name = "workspace.mlpab375647.airqmodelfdfb59"

try:
    model = client.get_registered_model(model_name)
    versions = client.search_model_versions(f"name='{model_name}'")

    results = {
        "model_name": model.name,
        "creation_timestamp": model.creation_timestamp,
        "versions": []
    }

    for v in versions:
        run = client.get_run(v.run_id)
        metrics = run.data.metrics
        results["versions"].append({
            "version": v.version,
            "run_id": v.run_id,
            "metrics": metrics
        })

    dbutils.notebook.exit(json.dumps(results))

except Exception as e:
    dbutils.notebook.exit(json.dumps({"error": str(e)[:500]}))
