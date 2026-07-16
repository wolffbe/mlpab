# Databricks notebook source
# COMMAND ----------
import json, requests

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()

results = {}

# The endpoint is at /api/2.0/postgres/synced_tables
# Try different structures for the synced_table field
# SYNCED_TABLE_ID = "workspace.mlpabf1452c.featuresb1ea93"

payloads = [
    {
        "synced_table": {
            "source_table": "workspace.mlpabf1452c.featuresb1ea93",
            "project_id": "mlpabf1452c-feat"
        }
    },
    {
        "synced_table": {
            "source_table_name": "workspace.mlpabf1452c.featuresb1ea93",
            "project": "mlpabf1452c-feat"
        }
    },
    {
        "synced_table": {
            "source": "workspace.mlpabf1452c.featuresb1ea93",
            "target_project": "mlpabf1452c-feat",
            "primary_keys": ["row_id"]
        }
    },
    {
        "synced_table": {
            "spec": {
                "source_table_full_name": "workspace.mlpabf1452c.featuresb1ea93",
                "project_id": "mlpabf1452c-feat",
                "primary_key_columns": ["row_id"]
            }
        }
    },
    {
        "synced_table": {
            "name": "workspace.mlpabf1452c.featuresb1ea93",
            "project_id": "mlpabf1452c-feat",
            "primary_key_columns": ["row_id"],
            "timeseries_key": "event_time"
        }
    },
]

for i, payload in enumerate(payloads):
    try:
        r = requests.post(
            f"{host}/api/2.0/postgres/synced_tables?synced_table_id=workspace.mlpabf1452c.featuresb1ea93",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        results[f'try_{i}'] = {"status": r.status_code, "body": r.text[:400]}
        if r.status_code in (200, 201):
            break
    except Exception as e:
        results[f'try_{i}'] = {"error": str(e)}

# Also look at what get-synced-table returns for existing synced tables
try:
    r_list = requests.get(
        f"{host}/api/2.0/postgres/synced_tables",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30
    )
    results['list_synced_tables'] = {"status": r_list.status_code, "body": r_list.text[:500]}
except Exception as e:
    results['list_synced_tables'] = {"error": str(e)}

dbutils.notebook.exit(json.dumps(results))
