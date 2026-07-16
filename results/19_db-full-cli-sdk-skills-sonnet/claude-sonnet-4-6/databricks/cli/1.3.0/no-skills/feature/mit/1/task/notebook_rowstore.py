# Databricks notebook source
# COMMAND ----------
import json, requests

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()

results = {}

# Try creating a "synced table" / row store table via the UC API
# This is the Databricks way to create an online store backed table

# Try using the Unity Catalog tables API to create a SYNCED TABLE
payload = {
    "name": "featuresb1ea93_online",
    "catalog_name": "workspace",
    "schema_name": "mlpabf1452c",
    "table_type": "EXTERNAL",
    "data_source_format": "DATABRICKS_FORMAT",
    "storage_location": "workspace.mlpabf1452c.featuresb1ea93"
}

try:
    r = requests.post(
        f"{host}/api/2.1/unity-catalog/tables",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=payload,
        timeout=30
    )
    results['tables_create'] = {"status": r.status_code, "body": r.text[:500]}
except Exception as e:
    results['tables_create'] = {"error": str(e)}

# Try the online store format
payload2 = {
    "name": "featuresb1ea93_online",
    "catalog_name": "workspace",
    "schema_name": "mlpabf1452c",
    "table_type": "EXTERNAL",
    "data_source_format": "DATABRICKS_ROW_STORE_FORMAT",
    "storage_location": f"s3://placeholder"
}

try:
    r2 = requests.post(
        f"{host}/api/2.1/unity-catalog/tables",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=payload2,
        timeout=30
    )
    results['tables_create_rowstore'] = {"status": r2.status_code, "body": r2.text[:500]}
except Exception as e:
    results['tables_create_rowstore'] = {"error": str(e)}

# Check the Unity Catalog tables endpoints
try:
    r3 = requests.get(
        f"{host}/api/2.1/unity-catalog/tables/workspace.mlpabf1452c.featuresb1ea93",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30
    )
    results['table_info'] = {"status": r3.status_code, "type": json.loads(r3.text).get("table_type", "N/A")}
except Exception as e:
    results['table_info'] = {"error": str(e)}

dbutils.notebook.exit(json.dumps(results))
