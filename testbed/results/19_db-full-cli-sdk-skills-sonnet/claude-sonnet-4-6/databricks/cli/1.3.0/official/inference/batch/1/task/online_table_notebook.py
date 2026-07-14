# Databricks notebook source
# COMMAND ----------
import requests
import json

host = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

# Try creating via REST API
url = f"https://{host}/api/2.0/online-tables"
payload = {
    "name": "workspace.mlpab6ef9cb.scores4f5893_online",
    "spec": {
        "source_table_full_name": "workspace.mlpab6ef9cb.scores4f5893",
        "primary_key_columns": ["account_id"],
        "run_triggered": {}
    }
}
r = requests.post(url, json=payload, headers=headers)
print(f"REST API: {r.status_code} - {r.text[:500]}")

# Also try SDK
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog as catalog_svc
w = WorkspaceClient()

# Check what methods are available
print(f"\nOnline tables methods: {[m for m in dir(w.online_tables) if not m.startswith('_')]}")

try:
    result = w.online_tables.create(name="workspace.mlpab6ef9cb.scores4f5893_online2",
                                     spec=catalog_svc.OnlineTableSpec(
                                         source_table_full_name="workspace.mlpab6ef9cb.scores4f5893",
                                         primary_key_columns=["account_id"],
                                         run_triggered=catalog_svc.OnlineTableSpecTriggeredSchedulingPolicy()
                                     ))
    dbutils.notebook.exit(f"Success: {result}")
except Exception as e:
    print(f"SDK error: {e}")
    dbutils.notebook.exit(f"REST: {r.status_code} {r.text[:300]}, SDK error: {str(e)[:300]}")
