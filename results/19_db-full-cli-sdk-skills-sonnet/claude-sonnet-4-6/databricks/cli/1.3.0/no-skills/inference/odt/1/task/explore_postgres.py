# Databricks notebook source
# COMMAND ----------
import json
import requests

host = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

results = {}

# Get details of the Postgres project
project_name = "mlpab08bf79-ccpred"

resp = requests.get(f"https://{host}/api/2.0/postgres/projects/{project_name}", headers=headers)
results["project_details"] = f"{resp.status_code}: {resp.text[:400]}"

# List branches
resp = requests.get(f"https://{host}/api/2.0/postgres/projects/{project_name}/branches", headers=headers)
results["branches"] = f"{resp.status_code}: {resp.text[:400]}"

# List compute endpoints
resp = requests.get(f"https://{host}/api/2.0/postgres/projects/{project_name}/compute-endpoints", headers=headers)
results["compute_endpoints"] = f"{resp.status_code}: {resp.text[:400]}"

# List roles
resp = requests.get(f"https://{host}/api/2.0/postgres/projects/{project_name}/roles", headers=headers)
results["roles"] = f"{resp.status_code}: {resp.text[:400]}"

dbutils.notebook.exit(json.dumps(results))
