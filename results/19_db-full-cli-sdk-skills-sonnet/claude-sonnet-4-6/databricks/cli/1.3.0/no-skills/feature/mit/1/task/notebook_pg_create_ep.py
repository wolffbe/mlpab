# Databricks notebook source
# COMMAND ----------
import json, requests

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()

results = {}

# Try creating a Postgres endpoint
# Based on existing endpoint properties
endpoint_payloads = [
    {
        "spec": {
            "autoscaling_limit_min_cu": 1,
            "autoscaling_limit_max_cu": 1,
            "suspend_timeout_duration": "86400s"
        }
    },
    {
        "endpoint_settings": {
            "autoscaling_limit_min_cu": 1,
            "autoscaling_limit_max_cu": 1,
            "suspend_timeout_duration": "86400s"
        }
    },
    {
        "compute": {
            "autoscaling_limit_min_cu": 1,
            "autoscaling_limit_max_cu": 1
        }
    },
    {
        "endpoint_id": "primary",
        "endpoint": {
            "autoscaling_limit_min_cu": 1,
            "autoscaling_limit_max_cu": 1
        }
    },
]

for i, payload in enumerate(endpoint_payloads):
    try:
        r = requests.post(
            f"{host}/api/2.0/postgres/projects/mlpabf1452c-feat/branches/production/endpoints?endpoint_id=primary",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        results[f'try_{i}'] = {"status": r.status_code, "body": r.text[:300]}
        if r.status_code == 200:
            break
    except Exception as e:
        results[f'try_{i}'] = {"error": str(e)}

# Also try the SDK approach
try:
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()
    # Check postgres-related methods
    postgres_attrs = [a for a in dir(w) if 'postgres' in a.lower() or 'lakebase' in a.lower()]
    results['sdk_postgres_attrs'] = postgres_attrs
except Exception as e:
    results['sdk_error'] = str(e)

dbutils.notebook.exit(json.dumps(results))
