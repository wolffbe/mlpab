# Databricks notebook source
# COMMAND ----------
import requests
import json

table_name = "workspace.mlpab0442b8.accountse81ff1"
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = spark.conf.get("spark.databricks.workspaceUrl")
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
base_url = f"https://{host}"

results = {}

# COMMAND ----------
# Try online tables endpoint
online_payload = {
    "name": table_name,
    "spec": {
        "source_table_full_name": table_name,
        "primary_key_columns": ["row_id"],
        "timeseries_key": "updated_at",
        "run_triggered": {}
    }
}
r1 = requests.post(f"{base_url}/api/2.0/online-tables", headers=headers, json=online_payload)
results["online_tables_status"] = r1.status_code
results["online_tables_response"] = r1.text[:500]

# COMMAND ----------
# Try synced database tables - triggered
synced_payload1 = {
    "name": table_name,
    "database_instance_name": "mlpab0442b8-lakebase",
    "logical_database_name": "mlpab0442b8",
    "spec": {
        "source_table_full_name": table_name,
        "primary_key_columns": ["row_id"],
        "run_triggered": {}
    }
}
r2 = requests.post(f"{base_url}/api/2.0/database/synced_tables", headers=headers, json=synced_payload1)
results["synced_triggered_status"] = r2.status_code
results["synced_triggered_response"] = r2.text[:500]

# COMMAND ----------
# Try synced database tables - continuous
synced_payload2 = {
    "name": table_name,
    "database_instance_name": "mlpab0442b8-lakebase",
    "logical_database_name": "mlpab0442b8",
    "spec": {
        "source_table_full_name": table_name,
        "primary_key_columns": ["row_id"],
        "run_continuously": {}
    }
}
r3 = requests.post(f"{base_url}/api/2.0/database/synced_tables", headers=headers, json=synced_payload2)
results["synced_continuous_status"] = r3.status_code
results["synced_continuous_response"] = r3.text[:500]

# COMMAND ----------
# Also try the mlpab0442b8db catalog
synced_payload3 = {
    "name": "mlpab0442b8db.mlpab0442b8.accountse81ff1",
    "database_instance_name": "mlpab0442b8-lakebase",
    "logical_database_name": "mlpab0442b8",
    "spec": {
        "source_table_full_name": table_name,
        "primary_key_columns": ["row_id"],
        "run_triggered": {}
    }
}
r4 = requests.post(f"{base_url}/api/2.0/database/synced_tables", headers=headers, json=synced_payload3)
results["synced_db_catalog_status"] = r4.status_code
results["synced_db_catalog_response"] = r4.text[:500]

# COMMAND ----------
# Get synced tables list (if available)
r5 = requests.get(f"{base_url}/api/2.0/database/synced_tables", headers=headers)
results["list_synced_status"] = r5.status_code
results["list_synced_response"] = r5.text[:200]

output = json.dumps(results, indent=2)
print(output)
dbutils.notebook.exit(output)
