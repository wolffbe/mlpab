# Databricks notebook source
# COMMAND ----------
import json
import requests

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
api_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
spark_url = spark.conf.get("spark.databricks.workspaceUrl")

results = {
    "api_url": api_url,
    "spark_url": spark_url,
}

# Try both URLs
source = "workspace.mlpaba35f2a.scored50223c"

for url_name, base in [("api_url", api_url), ("spark_url_https", f"https://{spark_url}")]:
    full_url = f"{base}/api/2.0/postgres/synced_tables?synced_table_id={source}"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"spec": {"source_table_full_name": source}}
    r = requests.post(full_url, headers=headers, json=payload, timeout=30)
    results[url_name] = f"{r.status_code}: {r.json().get('message', r.text[:200])}"

dbutils.notebook.exit(json.dumps(results))
