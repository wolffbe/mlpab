# Databricks notebook source
# COMMAND ----------
import json
import requests

host = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

results = {}

# Check Databricks workspace version
resp = requests.get(f"https://{host}/api/2.0/workspace-conf", headers=headers)
results["workspace_conf"] = f"{resp.status_code}: {resp.text[:200]}"

# Try to get the workspace version
resp = requests.get(f"https://{host}/api/2.0/clusters/spark-versions", headers=headers)
versions = resp.json()
results["latest_lts"] = [v for v in versions.get("versions", []) if "lts" in str(v).lower()][:3]

# Try to list all online tables in the workspace
resp = requests.get(f"https://{host}/api/2.0/online-tables?catalog_name=workspace", headers=headers)
results["online_tables_list"] = f"{resp.status_code}: {resp.text[:200]}"

# Try to see what synced-tables endpoints might look like
for path in [
    "/api/2.0/catalog/synced-tables?catalog_name=workspace&schema_name=mlpaba35f2a",
    "/api/2.0/synced-tables?catalog_name=workspace&schema_name=mlpaba35f2a",
    "/api/2.0/unity-catalog/tables?catalog_name=workspace&schema_name=mlpaba35f2a&table_type=SYNCED",
]:
    resp = requests.get(f"https://{host}{path}", headers=headers)
    results[path[:60]] = f"{resp.status_code}: {resp.text[:100]}"

dbutils.notebook.exit(json.dumps(results))
