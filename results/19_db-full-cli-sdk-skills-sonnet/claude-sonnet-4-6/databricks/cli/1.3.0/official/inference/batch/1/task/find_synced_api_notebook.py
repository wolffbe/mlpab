# Databricks notebook source
# COMMAND ----------
import requests

host = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

# Try various API paths for synced tables
paths = [
    ("POST", "/api/2.0/synced-tables"),
    ("POST", "/api/2.1/synced-tables"),
    ("GET", "/api/2.0/synced-tables"),
    ("POST", "/api/2.0/catalog/synced-tables"),
    ("POST", "/api/2.1/catalog/synced-tables"),
    ("POST", "/api/2.0/unity-catalog/synced-tables"),
    ("POST", "/api/2.1/unity-catalog/synced-tables"),
    ("POST", "/api/3.0/synced-tables"),
    ("POST", "/api/2.0/preview/synced-tables"),
]

payload = {
    "name": "workspace.mlpab6ef9cb.scores4f5893_synced",
    "spec": {
        "source_table_full_name": "workspace.mlpab6ef9cb.scores4f5893",
        "primary_key_columns": ["account_id"],
        "run_triggered": {}
    }
}

results = []
for method, path in paths:
    url = f"https://{host}{path}"
    if method == "POST":
        r = requests.post(url, json=payload, headers=headers)
    else:
        r = requests.get(url, headers=headers)
    status = r.status_code
    # Only show response text for non-404s
    text = r.text[:200] if status != 404 else ""
    results.append(f"{method} {path}: {status} {text}")

dbutils.notebook.exit("\n".join(results))
