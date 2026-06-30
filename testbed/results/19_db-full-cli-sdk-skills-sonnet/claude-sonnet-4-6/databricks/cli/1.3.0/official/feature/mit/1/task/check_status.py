# Databricks notebook source
import requests
import json

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

results = {}

# Check synced table status
url = f"{host}/api/2.0/database/synced_tables/workspace.mlpab9e4ddc.featuresb1ea93_online"
resp = requests.get(url, headers=headers, timeout=30)
results["get_synced_table"] = f"{resp.status_code}: {resp.text[:2000]}"

# Also check via Unity Catalog synced tables endpoint
url2 = f"{host}/api/2.1/unity-catalog/synced-tables/workspace.mlpab9e4ddc.featuresb1ea93_online"
resp2 = requests.get(url2, headers=headers, timeout=30)
results["get_uc_synced_table"] = f"{resp2.status_code}: {resp2.text[:2000]}"

dbutils.notebook.exit(json.dumps(results))
