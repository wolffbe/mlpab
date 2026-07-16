# Databricks notebook source
# MAGIC %md # Create Synced Table for Online Serving

# COMMAND ----------
import requests
import json
import os

CATALOG = "workspace"
SCHEMA = "mlpab3f803c"
PRED_TABLE = f"{CATALOG}.{SCHEMA}.ccpred739ee9"

# Get workspace URL from Spark config (available in serverless)
host = spark.conf.get("spark.databricks.workspaceUrl", "")
print(f"Host from spark conf: {host}")

# Get token from spark conf
token = spark.conf.get("spark.databricks.token", "")
if not token:
    # Try from secret scope or env var
    token = os.environ.get("DATABRICKS_TOKEN", "")

print(f"Token available: {bool(token)}")
print(f"Host: {host}")

# COMMAND ----------
if host and token:
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    # Try Synced Tables API
    synced_table_payload = {
        "name": f"{CATALOG}.{SCHEMA}.ccpred739ee9_synced",
        "spec": {
            "source_table_full_name": PRED_TABLE,
            "primary_key_columns": ["transaction_id"],
            "run_triggered": {
                "triggered_enable_continuous_auto_refresh": True
            }
        }
    }

    # Try different endpoints
    for url in [
        f"https://{host}/api/2.0/online-tables",
        f"https://{host}/api/2.0/synced-tables",
        f"https://{host}/api/2.1/synced-tables",
        f"https://{host}/api/2.0/catalog/synced-tables",
    ]:
        try:
            resp = requests.post(url, headers=headers, json=synced_table_payload, timeout=15)
            print(f"POST {url}: {resp.status_code}")
            print(f"Response: {resp.text[:300]}")
        except Exception as e:
            print(f"Error {url}: {e}")
else:
    print("No host or token available")

# COMMAND ----------
# Final verification
count = spark.table(PRED_TABLE).count()
print(f"ccpred739ee9 row count: {count}")
spark.table(PRED_TABLE).show(5)
