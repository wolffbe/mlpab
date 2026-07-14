# Databricks notebook source
# COMMAND ----------
# Create Synced Table for online/real-time access to scored50223c

from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as catalog

w = WorkspaceClient()

# Check if synced table already exists, if so delete it
try:
    existing = w.online_tables.get("workspace.mlpaba35f2a.scored50223c_online")
    print(f"Found existing online table: {existing}")
except Exception as e:
    print(f"No existing online table: {e}")

# Try to create via SDK - Synced Tables API
try:
    result = w.online_tables.create(
        name="workspace.mlpaba35f2a.scored50223c_online",
        spec=catalog.OnlineTableSpec(
            source_table_full_name="workspace.mlpaba35f2a.scored50223c",
            primary_key_columns=["request_id"],
            run_triggered=catalog.OnlineTableSpecTriggeredSchedulingPolicy()
        )
    )
    print(f"Online table created: {result}")
except Exception as e:
    print(f"Online table creation error: {e}")

    # Try Synced Tables approach
    try:
        import requests
        import os

        token = os.environ.get('DATABRICKS_TOKEN', dbutils.secrets.get('databricks', 'token') if False else None)
        host = spark.conf.get("spark.databricks.workspaceUrl")

        # Try Synced Tables endpoint
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        payload = {
            "name": "workspace.mlpaba35f2a.scored50223c_online",
            "spec": {
                "source_table_full_name": "workspace.mlpaba35f2a.scored50223c",
                "primary_key_columns": ["request_id"],
                "run_triggered": {}
            }
        }
        resp = requests.post(f"https://{host}/api/2.0/online-tables", json=payload, headers=headers)
        print(f"Response: {resp.status_code} {resp.text}")
    except Exception as e2:
        print(f"Error creating synced table: {e2}")
