# Databricks notebook source
# COMMAND ----------
import json, requests

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()

results = {}

# The structure is endpoint.spec.endpoint_type
# Try creating endpoint with the correct nested structure
endpoint_payloads = [
    {
        "endpoint": {
            "spec": {
                "endpoint_type": "ENDPOINT_TYPE_READ_WRITE",
                "autoscaling_limit_min_cu": 1,
                "autoscaling_limit_max_cu": 1,
                "suspend_timeout_duration": "86400s"
            }
        }
    },
    {
        "endpoint": {
            "spec": {
                "endpoint_type": "READ_WRITE",
                "autoscaling_limit_min_cu": 1,
                "autoscaling_limit_max_cu": 1
            }
        }
    },
    {
        "endpoint": {
            "endpoint_type": "ENDPOINT_TYPE_READ_WRITE",
            "autoscaling_limit_min_cu": 1,
            "autoscaling_limit_max_cu": 1,
            "suspend_timeout_duration": "86400s"
        }
    },
]

for i, payload in enumerate(endpoint_payloads):
    try:
        r = requests.post(
            f"{host}/api/2.0/postgres/projects/mlpabf1452c-feat/branches/production/endpoints?endpoint_id=primary",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=payload,
            timeout=60
        )
        results[f'try_{i}'] = {"status": r.status_code, "body": r.text[:500]}
        if r.status_code in (200, 201):
            break
    except Exception as e:
        results[f'try_{i}'] = {"error": str(e)}

dbutils.notebook.exit(json.dumps(results))
