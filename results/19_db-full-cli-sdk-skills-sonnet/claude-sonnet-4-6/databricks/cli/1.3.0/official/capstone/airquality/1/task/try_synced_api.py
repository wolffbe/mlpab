# Databricks notebook source

# COMMAND ----------
import json
results = {}

from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

# Try raw API calls to find synced tables endpoint
endpoints_to_try = [
    "GET /api/2.0/synced-tables",
    "GET /api/2.1/synced-tables",
    "GET /api/2.0/catalog/synced-tables",
    "GET /api/2.0/feature-tables/online-tables",
    "GET /api/2.0/feature-engineering/tables",
]

for ep in endpoints_to_try:
    method, path = ep.split(" ", 1)
    try:
        resp = w.api_client.do(method, path)
        results[ep] = "OK: " + str(resp)[:100]
    except Exception as e:
        results[ep] = f"ERR({type(e).__name__}): {str(e)[:100]}"

dbutils.notebook.exit(json.dumps(results))
