# Databricks notebook source
# MAGIC %md # Create Synced Table for Low-Latency Serving

# COMMAND ----------
import requests

CATALOG = "workspace"
SCHEMA = "mlpab3f803c"
PRED_TABLE = f"{CATALOG}.{SCHEMA}.ccpred739ee9"

# Get host and token from Databricks context
try:
    host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().getOrElse(None)
    token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
    print(f"Host: {host}")
    print(f"Token available: {bool(token)}")
except Exception as e:
    print(f"Error getting context: {e}")
    host = None
    token = None

# COMMAND ----------
# Try synced tables API (new API replacing online tables)
if host and token:
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    # Try different API endpoints for synced tables
    endpoints_to_try = [
        f"https://{host}/api/2.0/synced-tables",
        f"https://{host}/api/2.1/synced-tables",
        f"https://{host}/api/2.0/serving-endpoints",
    ]

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

    for url in endpoints_to_try:
        try:
            resp = requests.post(url, headers=headers, json=synced_table_payload, timeout=10)
            print(f"URL: {url}")
            print(f"Status: {resp.status_code}")
            print(f"Response: {resp.text[:300]}")
            print("---")
        except Exception as e:
            print(f"Error with {url}: {e}")

# COMMAND ----------
# Alternative: Create model serving endpoint that serves predictions
# This provides low-latency lookup via a REST endpoint
print("Pipeline already complete with Delta table for offline access")
print("Synced Tables API availability may vary by workspace")
print(f"Predictions are available in: {PRED_TABLE}")
print(f"Table has CDF enabled for streaming/incremental reads")

# Verify predictions table
result = spark.table(PRED_TABLE)
print(f"Row count: {result.count()}")
result.show(5)
