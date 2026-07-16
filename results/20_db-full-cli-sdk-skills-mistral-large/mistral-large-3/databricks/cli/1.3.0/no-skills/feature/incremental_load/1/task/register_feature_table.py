# Databricks notebook source
dbutils.library.installPyPI("databricks-feature-engineering", version="0.3.0")
dbutils.library.restartPython()

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

# Register the feature table
fe.create_table(
    name="${MLPAB_DATABRICKS_SCHEMA}.incrementala59b19",
    primary_keys=["row_id"],
    timestamp_keys=["event_time"],
    df=None,  # Use the existing table
    table_name="${MLPAB_DATABRICKS_SCHEMA}.incrementala59b19",
    description="Feature table for incremental load task"
)

# Enable online store
fe.publish_table(
    name="${MLPAB_DATABRICKS_SCHEMA}.incrementala59b19",
    online_store="databricks"
)