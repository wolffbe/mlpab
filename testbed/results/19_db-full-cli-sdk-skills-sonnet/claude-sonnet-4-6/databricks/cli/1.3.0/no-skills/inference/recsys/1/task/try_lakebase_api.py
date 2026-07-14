# Databricks notebook source
# COMMAND ----------
import requests
import json

host = spark.conf.get("spark.databricks.workspaceUrl", "")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

# COMMAND ----------
# First get the API spec for synced_tables POST
# Try GET to see the API shape
resp = requests.get(f"https://{host}/api/2.0/database/synced_tables", headers=headers)
print(f"GET synced_tables: {resp.status_code}: {resp.text[:300]}")

# COMMAND ----------
# Try different field names for the source table
payloads = [
    {
        "name": "workspace.mlpabb40f43.recs708df6_online",
        "database_instance_name": "mlpabb40f43-db",
        "spec": {
            "source_uc_table_name": "workspace.mlpabb40f43.recs708df6",
            "scheduling_policy": "TRIGGERED"
        }
    },
    {
        "name": "workspace.mlpabb40f43.recs708df6_online",
        "database_instance_name": "mlpabb40f43-db",
        "spec": {
            "delta_table_name": "workspace.mlpabb40f43.recs708df6",
            "scheduling_policy": "TRIGGERED"
        }
    },
    {
        "name": "workspace.mlpabb40f43.recs708df6_online",
        "database_instance_name": "mlpabb40f43-db",
        "spec": {
            "uc_table": "workspace.mlpabb40f43.recs708df6",
            "scheduling_policy": "TRIGGERED"
        }
    },
]

for payload in payloads:
    resp = requests.post(f"https://{host}/api/2.0/database/synced_tables", headers=headers, json=payload)
    field = list(payload["spec"].keys())[0]
    print(f"With spec.{field}: {resp.status_code}: {resp.text[:200]}")

# COMMAND ----------
# Also try with different source fields at top level
alt_payloads = [
    {
        "name": "workspace.mlpabb40f43.recs708df6_online",
        "database_instance_name": "mlpabb40f43-db",
        "source_table": "workspace.mlpabb40f43.recs708df6",
        "spec": {"scheduling_policy": "TRIGGERED"}
    },
    {
        "name": "workspace.mlpabb40f43.recs708df6_online",
        "database_instance_name": "mlpabb40f43-db",
        "spec": {
            "pipeline": {
                "source_table": "workspace.mlpabb40f43.recs708df6",
                "scheduling_policy": "TRIGGERED"
            }
        }
    },
]

for payload in alt_payloads:
    resp = requests.post(f"https://{host}/api/2.0/database/synced_tables", headers=headers, json=payload)
    print(f"Alt payload {list(payload.keys())}: {resp.status_code}: {resp.text[:200]}")

# COMMAND ----------
# Try to get API info from a get-table-spec type endpoint
for path in [
    "/api/2.0/database/synced_tables/spec",
    "/api/2.0/database/tables/spec",
    "/api/2.0/database",
]:
    resp = requests.get(f"https://{host}{path}", headers=headers)
    print(f"GET {path}: {resp.status_code}: {resp.text[:200]}")

dbutils.notebook.exit("done")
