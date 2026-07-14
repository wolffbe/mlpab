# Databricks notebook source
# COMMAND ----------
import requests
import json

host = spark.conf.get("spark.databricks.workspaceUrl", "")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

results = {}
table_name = "workspace.mlpabb40f43.recs708df6"

# COMMAND ----------
# Try various synced table API paths
synced_paths = [
    ("POST", "/api/2.0/uc/synced-tables"),
    ("POST", "/api/2.0/catalog/synced-tables"),
    ("POST", "/api/2.0/synced-tables/tables"),
    ("POST", "/api/2.1/synced-tables"),
    ("POST", "/api/2.0/delta-sync/tables"),
    ("POST", "/api/2.0/spark-sync/tables"),
    ("POST", "/api/2.0/online-tables-v2"),
    ("GET", "/api/2.0/unity-catalog/synced-tables"),
    ("GET", "/api/2.0/uc-online-tables"),
]

payload = {
    "name": "workspace.mlpabb40f43.recs708df6_synced",
    "spec": {
        "source_table_full_name": table_name,
        "primary_key_columns": ["rec_id"],
        "run_triggered": {}
    }
}

for method, path in synced_paths:
    if method == "POST":
        resp = requests.post(f"https://{host}{path}", headers=headers, json=payload)
    else:
        resp = requests.get(f"https://{host}{path}", headers=headers)
    results[f"{method} {path}"] = f"{resp.status_code}: {resp.text[:200]}"
    print(f"{method} {path}: {resp.status_code}: {resp.text[:150]}")

# COMMAND ----------
# Also try to set table properties for online access
try:
    spark.sql(f"""
        ALTER TABLE {table_name}
        SET TBLPROPERTIES (
            'delta.feature.appendOnly' = 'supported',
            'quality' = 'gold'
        )
    """)
    print("Table properties set")
except Exception as e:
    print(f"Error setting properties: {e}")

# COMMAND ----------
# Check if we can create a feature table through SQL (Databricks SQL extension)
try:
    # Try CREATE FEATURE TABLE syntax
    spark.sql(f"DESCRIBE EXTENDED {table_name}").show(50, truncate=False)
except Exception as e:
    print(f"Error: {e}")

dbutils.notebook.exit(json.dumps(results, indent=2))
