# Databricks notebook source
# MAGIC %md # Create Online Table via SDK

# COMMAND ----------
from databricks.sdk import WorkspaceClient
import json

CATALOG = "workspace"
SCHEMA = "mlpab3f803c"
PRED_TABLE = f"{CATALOG}.{SCHEMA}.ccpred739ee9"
ONLINE_TABLE = f"{CATALOG}.{SCHEMA}.ccpred739ee9_online"

print("Imports OK")

# COMMAND ----------
# Initialize workspace client (auto-detects credentials in serverless env)
w = WorkspaceClient()
host = w.config.host
print(f"Workspace: {host}")

# COMMAND ----------
# Try to create online table via raw API call
payload = {
    "name": ONLINE_TABLE,
    "spec": {
        "source_table_full_name": PRED_TABLE,
        "primary_key_columns": ["transaction_id"],
        "run_triggered": {
            "triggered_enable_continuous_auto_refresh": True
        }
    }
}

try:
    result = w.api_client.do("POST", "/api/2.0/online-tables", body=payload)
    print(f"Online table created: {result}")
except Exception as e:
    print(f"Online table creation failed: {e}")
    # Try synced tables path
    for path in ["/api/2.0/synced-tables", "/api/2.1/synced-tables"]:
        try:
            result = w.api_client.do("POST", path, body=payload)
            print(f"Synced table via {path}: {result}")
        except Exception as e2:
            print(f"Synced table {path} failed: {e2}")

# COMMAND ----------
# Final verification
count = spark.table(PRED_TABLE).count()
print(f"Predictions table rows: {count}")
