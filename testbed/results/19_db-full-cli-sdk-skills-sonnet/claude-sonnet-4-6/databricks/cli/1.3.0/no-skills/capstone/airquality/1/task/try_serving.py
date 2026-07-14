# Databricks notebook source

# COMMAND ----------

ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
token = ctx.apiToken().get()
host = ctx.apiUrl().get()

import requests, json
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

results = []

with open("/Volumes/workspace/mlpabd7768b/airqdata/serving_result.txt", "w") as f:
    # Try various API paths for synced tables
    paths = [
        ("POST", "/api/2.0/catalog/synced-tables", {"name": "workspace.mlpabd7768b.airqpredfdfb59_synced", "spec": {"source_table_full_name": "workspace.mlpabd7768b.airqpredfdfb59", "primary_key_columns": ["date"], "run_triggered": {}}}),
        ("POST", "/api/2.0/feature-engineering/synced-tables", {"name": "workspace.mlpabd7768b.airqpredfdfb59_synced", "spec": {"source_table_full_name": "workspace.mlpabd7768b.airqpredfdfb59", "primary_key_columns": ["date"], "run_triggered": {}}}),
    ]

    for method, path, body in paths:
        resp = requests.request(method, f"{host}{path}", headers=headers, json=body)
        f.write(f"{method} {path}: {resp.status_code}\n{resp.text[:300]}\n\n")

    # Try feature serving endpoint
    serving_config = {
        "name": "mlpabd7768b_airq_pred_serving",
        "config": {
            "served_entities": [
                {
                    "name": "airqpredfdfb59_entity",
                    "entity_version": "1",
                    "feature_spec_name": "workspace.mlpabd7768b.airqpredfdfb59_fspec",
                    "workload_size": "Small"
                }
            ]
        }
    }

    # Try to list serving endpoints
    resp = requests.get(f"{host}/api/2.0/serving-endpoints", headers=headers)
    f.write(f"GET /serving-endpoints: {resp.status_code}\n")
    if resp.ok:
        f.write(f"{resp.text[:500]}\n\n")

    # Try to create a serving endpoint for the predictions
    serving_endpoint_config = {
        "name": "mlpabd7768b-airqpred",
        "config": {
            "served_entities": []
        }
    }
    resp = requests.post(f"{host}/api/2.0/serving-endpoints", headers=headers, json=serving_endpoint_config)
    f.write(f"POST /serving-endpoints: {resp.status_code}\n{resp.text[:500]}\n")

print("Done - check serving_result.txt")
dbutils.notebook.exit("done")
