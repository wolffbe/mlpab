# Databricks notebook source
# COMMAND ----------
import json, requests

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()

results = {}

# Explore the Lakebase Postgres API to understand the endpoint creation
# First, list existing endpoints on one of the existing projects
try:
    r = requests.get(
        f"{host}/api/2.0/postgres/projects/mlpabefbb2e-feat/branches/production/endpoints",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30
    )
    results['list_existing_endpoints'] = {"status": r.status_code, "body": r.text[:1000]}
except Exception as e:
    results['list_existing_endpoints'] = {"error": str(e)}

# Try creating endpoint with minimal config
try:
    r2 = requests.post(
        f"{host}/api/2.0/postgres/projects/mlpabf1452c-feat/branches/production/endpoints/primary",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={
            "autoscaling_limit_min_cu": 0.25,
            "autoscaling_limit_max_cu": 1
        },
        timeout=30
    )
    results['create_endpoint'] = {"status": r2.status_code, "body": r2.text[:500]}
except Exception as e:
    results['create_endpoint'] = {"error": str(e)}

# Check the synced tables API
try:
    r3 = requests.get(
        f"{host}/api/2.0/postgres/projects/mlpabf1452c-feat/synced-tables",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30
    )
    results['list_synced_tables'] = {"status": r3.status_code, "body": r3.text[:500]}
except Exception as e:
    results['list_synced_tables'] = {"error": str(e)}

dbutils.notebook.exit(json.dumps(results))
