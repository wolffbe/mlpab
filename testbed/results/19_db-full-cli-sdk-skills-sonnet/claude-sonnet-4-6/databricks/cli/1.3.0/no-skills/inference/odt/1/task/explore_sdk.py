# Databricks notebook source
# COMMAND ----------
import json
import requests
from databricks.sdk import WorkspaceClient

host = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

results = {}

w = WorkspaceClient()

# List all methods/attributes on WorkspaceClient that contain online/synced/serving
attrs = [a for a in dir(w) if "online" in a.lower() or "synced" in a.lower() or "serving" in a.lower() or "feature" in a.lower()]
results["w_attrs"] = attrs

# Check databricks.sdk.service for synced tables
import databricks.sdk.service.catalog as cat
synced_classes = [x for x in dir(cat) if "synced" in x.lower()]
results["synced_classes"] = synced_classes

# Try raw API paths for Synced Tables
for method, path, data in [
    ("GET", "/api/2.0/online-tables", None),
    ("GET", "/api/2.0/synced-tables", None),
    ("GET", "/api/3.0/online-tables", None),
    ("GET", "/api/2.0/catalog/synced-tables", None),
    ("POST", "/api/2.0/synced-tables", {"name": "test", "spec": {}}),
]:
    if method == "GET":
        resp = requests.get(f"https://{host}{path}", headers=headers)
    else:
        resp = requests.post(f"https://{host}{path}", json=data, headers=headers)
    results[f"{method} {path}"] = f"{resp.status_code}: {resp.text[:100]}"

dbutils.notebook.exit(json.dumps(results))
