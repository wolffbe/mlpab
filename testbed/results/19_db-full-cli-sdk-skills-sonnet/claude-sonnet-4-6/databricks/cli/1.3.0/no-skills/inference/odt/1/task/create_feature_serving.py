# Databricks notebook source
# COMMAND ----------
import json
import requests

host = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

results = {}

# Try to create a Feature Serving endpoint
# This is different from a model serving endpoint - it serves Unity Catalog features directly
feature_serving_payload = {
    "name": "mlpaba35f2a_scored50223c_fs",
    "config": {
        "served_entities": [
            {
                "name": "scored50223c_features",
                "entity_name": "workspace.mlpaba35f2a.scored50223c",
                "entity_version": "1",
                "workload_size": "Small",
                "scale_to_zero_enabled": True
            }
        ]
    }
}

resp = requests.post(f"https://{host}/api/2.0/serving-endpoints", json=feature_serving_payload, headers=headers)
results["feature_serving_create"] = f"{resp.status_code}: {resp.text[:400]}"

# Try creating a Feature Store-style serving endpoint with a table entity
feature_spec_payload = {
    "name": "mlpaba35f2a_scored50223c_fs",
    "config": {
        "served_entities": [
            {
                "name": "scored50223c_entity",
                "feature_spec_name": "workspace.mlpaba35f2a.scored50223c",
                "workload_size": "Small",
                "scale_to_zero_enabled": True
            }
        ]
    }
}
resp = requests.post(f"https://{host}/api/2.0/serving-endpoints", json=feature_spec_payload, headers=headers)
results["feature_spec_serving"] = f"{resp.status_code}: {resp.text[:400]}"

dbutils.notebook.exit(json.dumps(results))
