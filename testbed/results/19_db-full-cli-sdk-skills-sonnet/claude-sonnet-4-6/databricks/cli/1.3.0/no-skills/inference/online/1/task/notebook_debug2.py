# Databricks notebook source
import json
import time
import requests

schema = "workspace.mlpabcef85c"
table_name = "profilesaa70e4"
full_table_name = f"{schema}.{table_name}"

ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
token = ctx.apiToken().get()
host = ctx.apiUrl().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

# COMMAND ----------
# Check Delta table exists and has CDF
print("=== Delta Table Check ===")
try:
    df = spark.table(full_table_name)
    df.show(3)
    print(f"Row count: {df.count()}")
except Exception as e:
    print(f"Table error: {e}")

spark.sql(f"SHOW TBLPROPERTIES {full_table_name}").show(truncate=False)

# COMMAND ----------
# Try to create online table - capture full error
print("=== Creating Online Table ===")
payload = {
    "name": full_table_name,
    "spec": {
        "source_table_full_name": full_table_name,
        "primary_key_columns": ["account_id"],
        "run_triggered": {}
    }
}
print(f"Payload: {json.dumps(payload, indent=2)}")

resp = requests.post(f"{host}/api/2.0/online-tables", headers=headers, json=payload)
print(f"Status: {resp.status_code}")
print(f"Response: {resp.text}")
print(f"Headers: {dict(resp.headers)}")

# COMMAND ----------
# Try GET on the online table
print("=== GET Online Table ===")
get_resp = requests.get(f"{host}/api/2.0/online-tables/{full_table_name}", headers=headers)
print(f"GET Status: {get_resp.status_code}")
print(f"GET Response: {get_resp.text}")

# COMMAND ----------
# Check what Unity Catalog table type it is
print("=== UC Table Details ===")
uc_resp = requests.get(
    f"{host}/api/2.1/unity-catalog/tables/{full_table_name}",
    headers=headers
)
print(f"UC Status: {uc_resp.status_code}")
uc_data = uc_resp.json()
print(f"table_type: {uc_data.get('table_type')}")
print(f"storage_location: {uc_data.get('storage_location')}")
print(f"data_source_format: {uc_data.get('data_source_format')}")
print(f"properties: {uc_data.get('properties', {})}")

# COMMAND ----------
# Try listing all online tables
print("=== List Online Tables ===")
list_resp = requests.get(f"{host}/api/2.0/online-tables", headers=headers)
print(f"List Status: {list_resp.status_code}")
print(f"List Response: {list_resp.text[:2000]}")

# COMMAND ----------
# Try a different online table name (not same as source)
print("=== Try separate online table name ===")
ot_name = f"{schema}.{table_name}_online"
payload2 = {
    "name": ot_name,
    "spec": {
        "source_table_full_name": full_table_name,
        "primary_key_columns": ["account_id"],
        "run_triggered": {}
    }
}
resp2 = requests.post(f"{host}/api/2.0/online-tables", headers=headers, json=payload2)
print(f"Status: {resp2.status_code}")
print(f"Response: {resp2.text}")

# COMMAND ----------
# Capture and return all debug info
debug_info = {
    "create_status": resp.status_code,
    "create_response": resp.text[:500],
    "get_status": get_resp.status_code,
    "get_response": get_resp.text[:500],
    "list_status": list_resp.status_code,
    "list_response": list_resp.text[:500],
    "create2_status": resp2.status_code,
    "create2_response": resp2.text[:500]
}
print(json.dumps(debug_info, indent=2))
dbutils.notebook.exit(json.dumps(debug_info))
