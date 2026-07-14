# Databricks notebook source
# COMMAND ----------
import json
import requests

host = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

results = {}

# Try different field structures for synced_table
# The synced_table likely needs the Postgres project/branch reference
variants = {
    "project_ref": {
        "synced_table": {
            "parent": "projects/mlpab08bf79-ccpred/branches/production/databases/databricks-postgres"
        }
    },
    "project_and_source": {
        "synced_table": {
            "parent": "projects/mlpab08bf79-ccpred",
            "source": "workspace.mlpaba35f2a.scored50223c"
        }
    },
    "branch_ref": {
        "synced_table": {
            "branch": "projects/mlpab08bf79-ccpred/branches/production"
        }
    },
    "project_id": {
        "synced_table": {
            "project_id": "mlpab08bf79-ccpred",
            "branch_id": "production"
        }
    },
    "lakebase_db": {
        "synced_table": {
            "lakebase_database": "projects/mlpab08bf79-ccpred/branches/production/databases/databricks-postgres",
            "source_table": "workspace.mlpaba35f2a.scored50223c"
        }
    },
    "db_ref_only": {
        "synced_table": {
            "database": "projects/mlpab08bf79-ccpred/branches/production/databases/databricks-postgres"
        }
    }
}

for name, body in variants.items():
    resp = requests.post(
        f"https://{host}/api/2.0/postgres/synced_tables?synced_table_id=workspace.mlpaba35f2a.scored50223c",
        json=body,
        headers=headers
    )
    results[name] = f"{resp.status_code}: {resp.text[:200]}"

dbutils.notebook.exit(json.dumps(results))
