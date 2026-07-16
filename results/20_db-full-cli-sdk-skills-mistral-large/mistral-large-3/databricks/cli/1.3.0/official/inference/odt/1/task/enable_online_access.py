# Databricks notebook source
# MAGIC %md
# MAGIC ## Enable Online Access for Feature Table
# MAGIC 
# MAGIC This notebook enables online access for the feature table `workspace.mlpab3f631d.scoreda4f6e2`.

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

# Enable online access
fe.publish_table(
    name="workspace.mlpab3f631d.scoreda4f6e2"
)

# Verify online access
print("Online access enabled for workspace.mlpab3f631d.scoreda4f6e2")