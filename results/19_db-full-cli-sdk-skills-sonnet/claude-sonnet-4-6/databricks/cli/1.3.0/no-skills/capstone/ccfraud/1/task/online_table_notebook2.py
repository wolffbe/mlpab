# Databricks notebook source
# MAGIC %md # Create Online/Synced Table

# COMMAND ----------
import requests
import json

CATALOG = "workspace"
SCHEMA = "mlpab3f803c"
PRED_TABLE = f"{CATALOG}.{SCHEMA}.ccpred739ee9"

# Get host and token from Databricks context
host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().getOrElse(None)
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
print(f"Host: {host}")

headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

# COMMAND ----------
# Try Synced Tables API (replacement for deprecated Online Tables)
results = {}

# Test endpoints
test_endpoints = [
    ("GET", f"https://{host}/api/2.0/online-tables/{CATALOG}.{SCHEMA}.ccpred739ee9_online", None),
    ("POST", f"https://{host}/api/2.0/synced-tables", {
        "name": f"{CATALOG}.{SCHEMA}.ccpred739ee9_synced",
        "spec": {
            "source_table_full_name": PRED_TABLE,
            "primary_key_columns": ["transaction_id"]
        }
    }),
    ("POST", f"https://{host}/api/2.1/synced-tables", {
        "name": f"{CATALOG}.{SCHEMA}.ccpred739ee9_synced",
        "spec": {
            "source_table_full_name": PRED_TABLE,
            "primary_key_columns": ["transaction_id"]
        }
    }),
]

for method, url, payload in test_endpoints:
    try:
        if method == "GET":
            resp = requests.get(url, headers=headers, timeout=10)
        else:
            resp = requests.post(url, headers=headers, json=payload, timeout=10)
        results[url] = {"status": resp.status_code, "response": resp.text[:200]}
        print(f"{method} {url}: {resp.status_code} - {resp.text[:200]}")
    except Exception as e:
        results[url] = {"error": str(e)}
        print(f"Error with {url}: {e}")

# COMMAND ----------
# If online/synced table creation fails, the predictions are still available
# via the Delta table with CDF enabled (streaming reads / incremental reads)
table_info = spark.sql(f"DESCRIBE EXTENDED {PRED_TABLE}").filter("col_name = 'Location'").collect()
print(f"Table info: {table_info}")

count = spark.table(PRED_TABLE).count()
print(f"Predictions count: {count}")

dbutils.notebook.exit(json.dumps({"status": "complete", "count": count, "api_results": results}))
