# Databricks notebook source
# MAGIC %md
# MAGIC # Create Online/Real-time Access for Feature Table

# COMMAND ----------

import requests
import json

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()

headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

catalog = "workspace"
schema = "mlpab312fe6"
source_table = f"{catalog}.{schema}.incremental3526e9"
online_name = f"{catalog}.{schema}.incremental3526e9_online"

# COMMAND ----------

# Try Databricks SDK first
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog as sdk_catalog

w = WorkspaceClient()

# Try to create online table via SDK
try:
    result = w.online_tables.create(
        name=online_name,
        spec=sdk_catalog.OnlineTableSpec(
            source_table_full_name=source_table,
            primary_key_columns=["row_id"],
            run_triggered=sdk_catalog.OnlineTableSpecTriggeredSchedulingPolicy()
        )
    )
    print(f"Online table created via SDK: {result}")
    dbutils.notebook.exit(f"SUCCESS_SDK: {online_name}")
except Exception as e:
    print(f"SDK online table creation failed: {e}")

# COMMAND ----------

# Try REST API endpoints for synced tables
endpoints = [
    ("/api/2.0/online-tables", {
        "name": online_name,
        "spec": {
            "source_table_full_name": source_table,
            "primary_key_columns": ["row_id"],
            "run_triggered": {}
        }
    }),
    ("/api/2.0/catalog/tables", {
        "name": "incremental3526e9_online",
        "catalog_name": catalog,
        "schema_name": schema,
        "table_type": "STREAMING",
        "data_source_format": "DELTA"
    }),
]

for endpoint, payload in endpoints:
    url = f"{host}{endpoint}"
    print(f"\nTrying POST {url}")
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    print(f"Status: {resp.status_code}, Response: {resp.text[:300]}")

# COMMAND ----------

# Check what API paths exist by doing GETs
check_endpoints = [
    "/api/2.0/online-tables",
    "/api/2.0/serving-endpoints",
    "/api/2.0/feature-store/",
    "/api/2.1/unity-catalog/tables",
]

for ep in check_endpoints:
    resp = requests.get(f"{host}{ep}", headers=headers, timeout=10)
    print(f"GET {ep}: {resp.status_code}")

dbutils.notebook.exit("EXPLORATION_DONE")
