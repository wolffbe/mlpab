# Databricks notebook source
# COMMAND ----------
import json, requests

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()

results = {}

# Find existing synced tables by checking other projects' tables
# First, list tables in the mlpabefbb2e schema
try:
    r = requests.get(
        f"{host}/api/2.1/unity-catalog/tables",
        headers={"Authorization": f"Bearer {token}"},
        params={"catalog_name": "workspace", "schema_name": "mlpabefbb2e"},
        timeout=30
    )
    results['mlpabefbb2e_tables'] = {"status": r.status_code, "body": r.text[:1000]}
except Exception as e:
    results['mlpabefbb2e_tables'] = {"error": str(e)}

# Try to get a synced table from existing project
# Let's check if there are any synced_tables on the feature online store project mlpabefbb2e-feat
try:
    r2 = requests.get(
        f"{host}/api/2.0/postgres/synced_tables/workspace.mlpabefbb2e.featuresb1ea93",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30
    )
    results['get_synced_efbb2e'] = {"status": r2.status_code, "body": r2.text[:500]}
except Exception as e:
    results['get_synced_efbb2e'] = {"error": str(e)}

# Let's check what schemas exist in workspace catalog (look for other runs)
try:
    r3 = requests.get(
        f"{host}/api/2.1/unity-catalog/schemas",
        headers={"Authorization": f"Bearer {token}"},
        params={"catalog_name": "workspace"},
        timeout=30
    )
    resp = r3.json()
    schemas = [s.get("name") for s in resp.get("schemas", [])]
    results['schemas'] = schemas[:20]
except Exception as e:
    results['schemas'] = str(e)

# Try creating synced table with project_id in the synced_table object
create_payloads = [
    {
        "synced_table": {
            "project_id": "mlpabf1452c-feat"
        }
    },
    {
        "synced_table": {
            "source_table_full_name": "workspace.mlpabf1452c.featuresb1ea93"
        }
    },
    {
        "synced_table": {
            "primary_key_columns": ["row_id"]
        }
    },
]

for i, payload in enumerate(create_payloads):
    try:
        r = requests.post(
            f"{host}/api/2.0/postgres/synced_tables?synced_table_id=workspace.mlpabf1452c.featuresb1ea93",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        results[f'create_{i}'] = {"status": r.status_code, "body": r.text[:500]}
        if r.status_code in (200, 201):
            break
    except Exception as e:
        results[f'create_{i}'] = {"error": str(e)}

dbutils.notebook.exit(json.dumps(results))
