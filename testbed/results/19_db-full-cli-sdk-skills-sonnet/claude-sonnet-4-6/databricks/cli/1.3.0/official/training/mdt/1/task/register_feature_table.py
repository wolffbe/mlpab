# Databricks notebook source
# COMMAND ----------
from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

# Read existing table
df = spark.table("workspace.mlpabc8d80a.scaled7ecfaf")
print(f"Rows: {df.count()}")
df.show(5)

# COMMAND ----------
# Drop existing table and recreate as feature table with primary key
spark.sql("DROP TABLE IF EXISTS workspace.mlpabc8d80a.scaled7ecfaf")

# Create feature table with proper primary key using Feature Engineering client
fe.create_table(
    name="workspace.mlpabc8d80a.scaled7ecfaf",
    primary_keys=["row_id"],
    df=df,
    schema=df.schema,
    description="Standardized feature table with training and serving splits"
)

print("Feature table registered successfully")

# COMMAND ----------
# Verify registration
ft = fe.get_table("workspace.mlpabc8d80a.scaled7ecfaf")
print(f"Feature table: {ft.name}")
print(f"Primary keys: {ft.primary_keys}")
print(f"Features: {ft.features}")
