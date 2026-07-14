# Databricks notebook source
# COMMAND ----------
# Try to publish feature table to online store using Feature Engineering client

import subprocess

# Check what's available
result = subprocess.run(['pip', 'show', 'databricks-feature-engineering'], capture_output=True, text=True)
print(result.stdout)
print(result.stderr)

# COMMAND ----------
# Try feature engineering approach
try:
    from databricks.feature_engineering import FeatureEngineeringClient
    print("Feature Engineering client available")

    fe = FeatureEngineeringClient()

    # Try to create/register feature table
    try:
        fe.create_table(
            name="workspace.mlpabb40f43.recs708df6",
            primary_keys=["rec_id"],
            schema=spark.table("workspace.mlpabb40f43.recs708df6").schema,
            description="Top-5 recommendations per user"
        )
        print("Feature table created")
    except Exception as e:
        print(f"Create table error (may already exist): {e}")

    print("Feature Engineering client initialized successfully")

except ImportError as e:
    print(f"Feature Engineering not available: {e}")

# COMMAND ----------
# Try to create online store / synced table using REST API directly
import requests
import os

host = spark.conf.get("spark.databricks.workspaceUrl", "")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

print(f"Host: {host}")

# Try synced tables API
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

# Try various API paths for synced tables
api_paths = [
    "/api/2.0/feature-store/tables",
    "/api/2.1/unity-catalog/online-tables",
    "/api/2.0/serving-endpoints",
]

for path in api_paths:
    try:
        resp = requests.get(f"https://{host}{path}", headers=headers)
        print(f"GET {path}: {resp.status_code}")
        if resp.status_code == 200:
            print(resp.text[:200])
    except Exception as e:
        print(f"Error for {path}: {e}")

# COMMAND ----------
# Create Feature Serving Endpoint for online access (alternative to Online Table)
import json

fe_endpoint_payload = {
    "name": "mlpabb40f43_recs708df6_endpoint",
    "config": {
        "served_entities": [
            {
                "feature_function_name": "workspace.mlpabb40f43.recs708df6",
                "entity_lookup_enabled": True
            }
        ]
    }
}

# Or use serving-endpoints to create a feature serving endpoint
host = spark.conf.get("spark.databricks.workspaceUrl", "")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

# Check if feature serving endpoint API is available
resp = requests.get(f"https://{host}/api/2.0/serving-endpoints", headers=headers)
print(f"Serving endpoints status: {resp.status_code}")
if resp.status_code == 200:
    data = resp.json()
    print(f"Existing endpoints: {[e.get('name') for e in data.get('endpoints', [])]}")

# COMMAND ----------
# Try Feature Engineering publish approach
try:
    from databricks.feature_engineering import FeatureEngineeringClient
    from databricks.feature_engineering.entities.feature_serving_endpoint import (
        EndpointCoreConfig,
        ServedEntity,
    )

    fe = FeatureEngineeringClient()

    # Create feature serving endpoint for online access
    endpoint_name = "mlpabb40f43_recs708df6_endpoint"

    try:
        fe.create_feature_serving_endpoint(
            name=endpoint_name,
            config=EndpointCoreConfig(
                served_entities=ServedEntity(
                    feature_spec_name="workspace.mlpabb40f43.recs708df6",
                    workload_size="Small",
                    scale_to_zero_enabled=True
                )
            )
        )
        print(f"Feature serving endpoint {endpoint_name} created")
    except Exception as e:
        print(f"Error creating feature serving endpoint: {e}")

except Exception as e:
    print(f"Error: {e}")
