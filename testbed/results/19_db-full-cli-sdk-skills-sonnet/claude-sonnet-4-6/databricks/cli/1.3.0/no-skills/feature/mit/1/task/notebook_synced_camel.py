# Databricks notebook source
# COMMAND ----------
import json, requests

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()

results = {}

# Try camelCase field names (protobuf JSON uses camelCase)
create_payloads = [
    # camelCase versions
    {
        "syncedTable": {
            "projectId": "mlpabf1452c-feat"
        }
    },
    {
        "synced_table": {
            "projectId": "mlpabf1452c-feat"
        }
    },
    # Try without wrapper
    {
        "projectId": "mlpabf1452c-feat",
        "sourceTableFullName": "workspace.mlpabf1452c.featuresb1ea93"
    },
    # The Lakebase Postgres create synced table might use different struct
    {
        "synced_table": {
            "sourceTableFullName": "workspace.mlpabf1452c.featuresb1ea93"
        }
    },
    {
        "synced_table": {
            "projectId": "mlpabf1452c-feat",
            "sourceTableFullName": "workspace.mlpabf1452c.featuresb1ea93",
            "primaryKeyColumns": ["row_id"],
            "timeseriesKey": "event_time"
        }
    },
    # Try with just sourceTableFullName
    {
        "synced_table": {
            "sourceTable": "workspace.mlpabf1452c.featuresb1ea93",
            "project": "projects/mlpabf1452c-feat"
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
        results[f'try_{i}'] = {"status": r.status_code, "body": r.text[:400]}
        if r.status_code in (200, 201):
            break
    except Exception as e:
        results[f'try_{i}'] = {"error": str(e)}

dbutils.notebook.exit(json.dumps(results))
