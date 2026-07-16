# Databricks notebook source
# COMMAND ----------
# Try to set up online feature store access for accountse81ff1

table_name = "workspace.mlpab0442b8.accountse81ff1"

# COMMAND ----------
# Check current state of the table
df = spark.table(table_name)
print(f"Row count: {df.count()}")
df.show(5)

# COMMAND ----------
# Try Feature Engineering client first
try:
    from databricks.feature_engineering import FeatureEngineeringClient
    fe = FeatureEngineeringClient()

    # Check if table is registered as feature table
    try:
        ft = fe.get_table(name=table_name)
        print(f"Feature table exists: {ft}")
    except Exception as e:
        print(f"Table not registered, registering: {e}")
        fe.register_table(
            name=table_name,
            primary_keys=["row_id"],
            timestamp_keys=["updated_at"],
            description="Accounts feature table - latest revision per row_id"
        )
        print(f"Registered feature table: {table_name}")

    # Try to publish to online store (using Synced Tables / Online Tables)
    try:
        from databricks.feature_engineering.entities.feature_serving_endpoint import ServedEntity
        # Try creating a feature serving endpoint
        print("Attempting to create feature serving endpoint...")
        fe.create_feature_serving_endpoint(
            name="mlpab0442b8-accountse81ff1",
            config=EndpointCoreConfigInput(
                served_entities=[
                    ServedEntity(
                        feature_spec_name=table_name,
                        workload_size="Small",
                        scale_to_zero_enabled=True,
                    )
                ]
            )
        )
        print("Feature serving endpoint created")
    except Exception as e:
        print(f"Feature serving endpoint error: {e}")

    # Try online store publish
    try:
        from databricks.feature_engineering import FeatureLookup
        fe.publish_table(
            name=table_name,
            online_store=OnlineStoreSpec(online_store_type=DatabricksOnlineStoreSpec(
                source_data_specifications=SourceDataSpecificationsInput(
                    databricks_workspace_ids=None
                )
            ))
        )
        print("Published to online store")
    except Exception as e:
        print(f"Publish error: {e}")

except ImportError as e:
    print(f"Feature Engineering import error: {e}")

# COMMAND ----------
# Try alternative: create online table via DatabricksClient REST API
import requests
import json
import os

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = spark.conf.get("spark.databricks.workspaceUrl")

headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

# Try the online tables endpoint
online_table_spec = {
    "name": table_name,
    "spec": {
        "source_table_full_name": table_name,
        "primary_key_columns": ["row_id"],
        "timeseries_key": "updated_at",
        "run_triggered": {}
    }
}

response = requests.post(
    f"https://{host}/api/2.0/online-tables",
    headers=headers,
    json=online_table_spec
)
print(f"Online tables API response: {response.status_code}")
print(response.text[:500])

# COMMAND ----------
# Try synced database tables endpoint
synced_table_spec = {
    "name": table_name,
    "database_instance_name": "mlpab0442b8-lakebase",
    "logical_database_name": "mlpab0442b8",
    "spec": {
        "source_table_full_name": table_name,
        "primary_key_columns": ["row_id"],
        "run_triggered": {"pipeline_type": "TRIGGERED"}
    }
}

response2 = requests.post(
    f"https://{host}/api/2.0/database/synced_tables",
    headers=headers,
    json=synced_table_spec
)
print(f"Synced tables API response: {response2.status_code}")
print(response2.text[:1000])

print("Done exploring online store options")
