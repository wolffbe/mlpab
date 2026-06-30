# Databricks notebook source

# COMMAND ----------
import requests
import json

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

results = {}

# Try creating an online table
url = f"{host}/api/2.0/online-tables"
payload = {
    "name": "workspace.mlpab9e4ddc.featuresb1ea93_online",
    "spec": {
        "source_table_full_name": "workspace.mlpab9e4ddc.featuresb1ea93",
        "primary_key_columns": ["row_id"],
        "timeseries_key": "event_time",
        "run_triggered": {}
    }
}
resp = requests.post(url, headers=headers, json=payload)
results["online_table_status"] = resp.status_code
results["online_table_response"] = resp.text[:2000]

# Try SDK
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
attrs = [a for a in dir(w) if not a.startswith('_')]
online_attrs = [a for a in attrs if 'online' in a.lower() or 'synced' in a.lower() or 'feature' in a.lower()]
results["sdk_online_attrs"] = online_attrs

# Try SDK online_tables
if hasattr(w, 'online_tables'):
    try:
        from databricks.sdk.service.catalog import OnlineTableSpec, PrimaryKeyConstraint
        result = w.online_tables.create(
            name="workspace.mlpab9e4ddc.featuresb1ea93_online",
            spec=OnlineTableSpec(
                source_table_full_name="workspace.mlpab9e4ddc.featuresb1ea93",
                primary_key_columns=["row_id"],
                timeseries_key="event_time",
                run_triggered={}
            )
        )
        results["sdk_create"] = str(result)
    except Exception as e:
        results["sdk_create_error"] = str(e)

dbutils.notebook.exit(json.dumps(results))
