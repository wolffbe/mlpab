# Databricks notebook source
# COMMAND ----------
import pandas as pd
from databricks.feature_engineering import FeatureEngineeringClient

# COMMAND ----------
# Read predictions from volume
volume_path = "/Volumes/workspace/mlpab5c18ba/mlpab5c18ba_data"
preds = pd.read_csv(f"{volume_path}/predictions.csv")
print(f"Predictions shape: {preds.shape}")
print(preds.head())

# COMMAND ----------
# Create the feature table using Feature Engineering client
fe = FeatureEngineeringClient()

# Drop existing table if any
try:
    spark.sql("DROP TABLE IF EXISTS workspace.mlpab5c18ba.predictions7b586d")
    print("Dropped existing table")
except Exception as e:
    print(f"Drop failed (ok): {e}")

# Create feature table
preds_spark = spark.createDataFrame(preds)

feature_table = fe.create_table(
    name="workspace.mlpab5c18ba.predictions7b586d",
    primary_keys=["row_id"],
    df=preds_spark,
    description="Predictions from trainjob7b586d logistic regression model"
)
print(f"Feature table created: {feature_table}")

# COMMAND ----------
# Verify the feature table
ft = fe.get_table(name="workspace.mlpab5c18ba.predictions7b586d")
print(f"Feature table: {ft.name}")
print(f"Primary keys: {ft.primary_keys}")
print(f"Features: {ft.features}")

# COMMAND ----------
print("Feature table setup complete")
