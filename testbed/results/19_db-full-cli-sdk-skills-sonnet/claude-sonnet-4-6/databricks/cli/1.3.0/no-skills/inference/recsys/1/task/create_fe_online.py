# Databricks notebook source
# COMMAND ----------
# Use Feature Engineering client to create feature table with online access

import json
import requests

host = spark.conf.get("spark.databricks.workspaceUrl", "")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

# COMMAND ----------
# Check Feature Engineering client methods
from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

print("Available methods:")
for method in sorted(dir(fe)):
    if not method.startswith('_'):
        print(f"  {method}")

# COMMAND ----------
# Create the feature table using FE client (may already exist)
table_name = "workspace.mlpabb40f43.recs708df6"
recs_df = spark.table(table_name)

try:
    ft = fe.create_table(
        name=table_name,
        primary_keys=["rec_id"],
        schema=recs_df.schema,
        description="Top-5 recommendations per user"
    )
    print(f"Feature table created: {ft}")
except Exception as e:
    print(f"Create table result: {e}")

# Write data to feature table
try:
    fe.write_table(
        name=table_name,
        df=recs_df,
        mode="overwrite"
    )
    print("Feature table data written")
except Exception as e:
    print(f"Write table result: {e}")

# COMMAND ----------
# Try to publish to online store using FE client
try:
    from databricks.feature_engineering import FeatureEngineeringClient

    fe = FeatureEngineeringClient()

    # Check if publish_table method exists
    if hasattr(fe, 'publish_table'):
        print("publish_table method exists")

        # Try to publish using the online store API
        # First, check what online store specs are available
        import inspect
        sig = inspect.signature(fe.publish_table)
        print(f"publish_table signature: {sig}")
    else:
        print("No publish_table method")

    # Check for online-related methods
    online_methods = [m for m in dir(fe) if 'online' in m.lower()]
    print(f"Online-related methods: {online_methods}")

except Exception as e:
    print(f"Error: {e}")

# COMMAND ----------
# Try to create feature serving endpoint via REST API
# Check what features spec is available
endpoint_name = "mlpabb40f43_recs_serving"

# Try feature store REST API endpoint
resp = requests.get(f"https://{host}/api/2.0/feature-store/tables/{table_name}", headers=headers)
print(f"GET feature-store table: {resp.status_code} - {resp.text[:200]}")

# Try to get feature store tables list
resp = requests.get(f"https://{host}/api/2.0/feature-store/tables", headers=headers)
print(f"GET feature-store tables: {resp.status_code} - {resp.text[:200]}")

# Try feature specs
resp = requests.get(f"https://{host}/api/2.0/feature-store/feature-specs", headers=headers)
print(f"GET feature-specs: {resp.status_code} - {resp.text[:200]}")

# COMMAND ----------
# Try creating online feature store endpoint using serving-endpoints API
payload = {
    "name": endpoint_name,
    "config": {
        "served_entities": [
            {
                "entity_name": "workspace.mlpabb40f43.recs708df6",
                "entity_version": "1",
                "scale_to_zero_enabled": True,
                "workload_size": "Small"
            }
        ]
    }
}

resp = requests.post(f"https://{host}/api/2.0/serving-endpoints", headers=headers, json=payload)
print(f"POST serving-endpoint: {resp.status_code}")
print(f"Response: {resp.text[:500]}")

dbutils.notebook.exit(json.dumps({
    "host": host,
    "status": "explored"
}))
