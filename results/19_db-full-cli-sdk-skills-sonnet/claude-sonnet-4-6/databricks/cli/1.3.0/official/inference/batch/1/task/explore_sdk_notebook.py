# Databricks notebook source
# COMMAND ----------
import requests
import json

host = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

# Try different API paths for synced tables
paths_to_try = [
    "/api/2.0/synced-tables",
    "/api/2.1/synced-tables",
    "/api/2.0/catalog/synced-tables",
    "/api/2.1/catalog/synced-tables",
    "/api/2.0/unity-catalog/synced-tables",
]

results = []
for path in paths_to_try:
    url = f"https://{host}{path}"
    r = requests.get(url, headers=headers)
    results.append(f"{path}: {r.status_code}")

output = "\n".join(results)
print(output)

# Also try to check SDK methods
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
sdk_attrs = [attr for attr in dir(w) if 'synced' in attr.lower() or 'online' in attr.lower()]
print(f"\nSDK methods: {sdk_attrs}")

dbutils.notebook.exit(output + "\nSDK:" + str(sdk_attrs))
