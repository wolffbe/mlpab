# Databricks notebook source
# COMMAND ----------
import requests
import json
import databricks.sdk.service.catalog as cat

host = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

results = {}

# Find online-table related classes
online_classes = [x for x in dir(cat) if "online" in x.lower() or "synced" in x.lower()]
results["online_classes"] = online_classes

# Try various API paths for online/synced tables
for path in [
    "/api/2.0/online-tables",
    "/api/2.0/synced-tables",
]:
    resp = requests.get(f"https://{host}{path}", headers=headers)
    results[path] = f"{resp.status_code}: {resp.text[:200]}"

# Check SDK online_tables service
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
results["online_tables_type"] = str(type(w.online_tables))
results["online_tables_methods"] = [m for m in dir(w.online_tables) if not m.startswith('_')]

dbutils.notebook.exit(json.dumps(results))
