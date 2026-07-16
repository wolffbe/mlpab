# Databricks notebook source
# COMMAND ----------
import json, requests

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()

results = {}

# Try creating synced table for featuresb1ea93_online
# Based on the existing synced table structure, it's a FOREIGN table backed by Postgres

create_payloads = [
    # Try with full project path
    {
        "synced_table": {
            "project": "projects/mlpabf1452c-feat"
        }
    },
    {
        "synced_table": {
            "project_path": "projects/mlpabf1452c-feat"
        }
    },
    # Try with database_name
    {
        "synced_table": {
            "database_name": "mlpabf1452c"
        }
    },
    # Try simple one-field with project_id
    {
        "synced_table": {
            "project_id": "projects/mlpabf1452c-feat"
        }
    },
    # Try source_table as field
    {
        "synced_table": {
            "source_table": {
                "name": "workspace.mlpabf1452c.featuresb1ea93"
            }
        }
    },
    # Try just table_name
    {
        "synced_table": {
            "table_name": "workspace.mlpabf1452c.featuresb1ea93"
        }
    },
]

for i, payload in enumerate(create_payloads):
    try:
        r = requests.post(
            f"{host}/api/2.0/postgres/synced_tables?synced_table_id=workspace.mlpabf1452c.featuresb1ea93_online",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        results[f'try_{i}'] = {"status": r.status_code, "body": r.text[:500]}
        if r.status_code in (200, 201):
            break
    except Exception as e:
        results[f'try_{i}'] = {"error": str(e)}

# Also try looking at the endpoint we created
try:
    r_ep = requests.get(
        f"{host}/api/2.0/postgres/projects/mlpabf1452c-feat/branches/production/endpoints/primary",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30
    )
    results['endpoint'] = {"status": r_ep.status_code, "body": r_ep.text[:500]}
except Exception as e:
    results['endpoint'] = {"error": str(e)}

dbutils.notebook.exit(json.dumps(results))
