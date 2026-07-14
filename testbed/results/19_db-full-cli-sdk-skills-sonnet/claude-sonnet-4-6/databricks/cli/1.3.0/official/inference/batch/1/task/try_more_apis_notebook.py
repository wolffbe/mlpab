# Databricks notebook source
# COMMAND ----------
import requests, json

host = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

lines = []

# Try more API paths for synced tables
payload = {
    "name": "workspace.mlpab6ef9cb.scores4f5893_synced",
    "spec": {
        "source_table_full_name": "workspace.mlpab6ef9cb.scores4f5893",
        "primary_key_columns": ["account_id"],
        "run_triggered": {}
    }
}

more_paths = [
    "/api/2.0/preview/unity-catalog/synced-tables",
    "/api/2.1/preview/unity-catalog/synced-tables",
    "/api/2.0/catalog/tables/workspace.mlpab6ef9cb.scores4f5893/online",
    "/api/2.0/unity-catalog/tables/workspace.mlpab6ef9cb.scores4f5893/online",
]

for path in more_paths:
    r = requests.post(f"https://{host}{path}", json=payload, headers=headers)
    lines.append(f"POST {path}: {r.status_code} {r.text[:150] if r.status_code != 404 else ''}")

# Also check if there's a current list of synced tables by GET
list_paths = [
    "/api/2.0/synced-tables/",
    "/api/2.0/online-tables/workspace.mlpab6ef9cb.scores4f5893",
]
for path in list_paths:
    r = requests.get(f"https://{host}{path}", headers=headers)
    lines.append(f"GET {path}: {r.status_code} {r.text[:150] if r.status_code != 404 else ''}")

# Let's also try to update the table to add an online-layer
patch_payload = {
    "name": "workspace.mlpab6ef9cb.scores4f5893_synced",
    "spec": {
        "source_table_full_name": "workspace.mlpab6ef9cb.scores4f5893",
        "primary_key_columns": ["account_id"],
    }
}
r = requests.put(f"https://{host}/api/2.0/online-tables/workspace.mlpab6ef9cb.scores4f5893_synced", json=patch_payload, headers=headers)
lines.append(f"PUT online-tables: {r.status_code} {r.text[:200] if r.status_code != 404 else ''}")

dbutils.notebook.exit("\n".join(lines))
