# Databricks notebook source
# COMMAND ----------
import json
import requests

host = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

results = {}

# Try different variations of creating a Synced Table
payloads = [
    {
        "path": "/api/2.0/online-tables",
        "body": {
            "name": "workspace.mlpaba35f2a.scored50223c_online",
            "spec": {
                "source_table_full_name": "workspace.mlpaba35f2a.scored50223c",
                "primary_key_columns": ["request_id"],
                "run_continuously": {}
            }
        }
    },
    {
        "path": "/api/2.0/online-tables",
        "body": {
            "name": "workspace.mlpaba35f2a.scored50223c_online",
            "spec": {
                "source_table_full_name": "workspace.mlpaba35f2a.scored50223c",
                "primary_key_columns": ["request_id"],
                "run_triggered": {"triggered_updates": {"checkpoint_location": "dbfs:/tmp"}}
            },
            "type": "SYNCED"
        }
    }
]

for item in payloads:
    resp = requests.post(f"https://{host}{item['path']}", json=item['body'], headers=headers)
    results[f"POST {item['path']} - {list(item['body'].get('spec', {}).keys())}"] = f"{resp.status_code}: {resp.text[:300]}"

# Try Unity Catalog table creation with SYNCED type
uc_payload = {
    "name": "scored50223c_synced",
    "catalog_name": "workspace",
    "schema_name": "mlpaba35f2a",
    "table_type": "SYNCED",
    "data_source_format": "DELTA",
    "properties": {
        "source_table": "workspace.mlpaba35f2a.scored50223c",
        "primary_key": "request_id"
    }
}
resp = requests.post(f"https://{host}/api/2.1/unity-catalog/tables", json=uc_payload, headers=headers)
results["UC SYNCED table"] = f"{resp.status_code}: {resp.text[:300]}"

# Check tables API for table types
resp = requests.get(f"https://{host}/api/2.0/unity-catalog/tables?catalog_name=workspace&schema_name=mlpaba35f2a", headers=headers)
results["existing_tables"] = resp.json().get("tables", [{}])[0].get("name", "none")

dbutils.notebook.exit(json.dumps(results))
