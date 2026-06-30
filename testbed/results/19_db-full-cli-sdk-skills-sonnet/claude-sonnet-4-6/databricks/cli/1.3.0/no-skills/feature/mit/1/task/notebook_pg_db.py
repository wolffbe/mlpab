# Databricks notebook source
# COMMAND ----------
import json, requests

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()

results = {}

# First check what databases exist on the existing project mlpabefbb2e-feat
try:
    r = requests.get(
        f"{host}/api/2.0/postgres/projects/mlpabefbb2e-feat/branches/production/databases",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30
    )
    results['list_dbs'] = {"status": r.status_code, "body": r.text[:1000]}
except Exception as e:
    results['list_dbs'] = {"error": str(e)}

# Try creating a database with various payloads
# The 'database' wrapper seems to need specific fields
db_payloads = [
    {"database": {"pg_version": 17}},
    {"database": {"settings": {}}},
    {"database": {"schemas": [{"name": "public"}]}},
    {"database": {"encoding": "UTF8"}},
    {"database": {"collation": "en_US.UTF-8"}},
]

for i, payload in enumerate(db_payloads):
    try:
        r = requests.post(
            f"{host}/api/2.0/postgres/projects/mlpabf1452c-feat/branches/production/databases?database_id=mlpabf1452c",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        results[f'create_db_{i}'] = {"status": r.status_code, "body": r.text[:300]}
        if r.status_code in (200, 201):
            break
    except Exception as e:
        results[f'create_db_{i}'] = {"error": str(e)}

# Try synced table with database field in spec
synced_payloads = [
    {
        "spec": {
            "source_table_full_name": "workspace.mlpabf1452c.featuresb1ea93",
            "database": "projects/mlpabf1452c-feat/branches/production/databases/mlpabf1452c"
        }
    },
    {
        "spec": {
            "source_table_full_name": "workspace.mlpabf1452c.featuresb1ea93",
            "database": "mlpabf1452c"
        }
    },
    {
        "spec": {
            "source_table_full_name": "workspace.mlpabf1452c.featuresb1ea93",
            "database_name": "mlpabf1452c",
            "project_id": "mlpabf1452c-feat"
        }
    },
]

for i, payload in enumerate(synced_payloads):
    try:
        r = requests.post(
            f"{host}/api/2.0/postgres/synced_tables?synced_table_id=workspace.mlpabf1452c.featuresb1ea93_online",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        results[f'synced_{i}'] = {"status": r.status_code, "body": r.text[:400]}
        if r.status_code in (200, 201):
            break
    except Exception as e:
        results[f'synced_{i}'] = {"error": str(e)}

dbutils.notebook.exit(json.dumps(results))
