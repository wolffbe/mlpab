# Databricks notebook source
# COMMAND ----------
# Try creating Lakebase instance for online serving

import json, requests
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
host = w.config.host.rstrip('/')
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

results = {}

# Create a Lakebase Postgres database instance for online serving
lakebase_payload = {
    "name": "mlpab62f111-online-store",
    "storage_type": "SERVERLESS"
}

r = requests.post(f"{host}/api/2.0/database/instances", headers=headers, json=lakebase_payload, timeout=30)
results['create_lakebase'] = {"status_code": r.status_code, "response": r.text[:500]}

# Also try fetching all tables via the catalog explorer API to understand available endpoints
r = requests.get(f"{host}/api/2.0/unity-catalog/tables?catalog_name=workspace&schema_name=mlpab62f111", headers=headers, timeout=10)
results['tables_list'] = {"status_code": r.status_code, "response": r.text[:500]}

dbutils.notebook.exit(json.dumps(results))
