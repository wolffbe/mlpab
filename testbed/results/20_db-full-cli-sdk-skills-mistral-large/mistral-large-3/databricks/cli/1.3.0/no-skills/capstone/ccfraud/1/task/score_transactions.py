# Databricks notebook source
# MAGIC %md
# MAGIC ## Score Transactions for Fraud Probability

# COMMAND ----------

from pyspark.sql import SparkSession
import mlflow
import mlflow.spark

# Load model
model_uri = "models:/ccmodele0b071/latest"
model = mlflow.spark.load_model(model_uri)

# Load scoring data
df_score = spark.table("workspace.${MLPAB_DATABRICKS_SCHEMA}.score_transactions")

# Score
predictions = model.transform(df_score)

# Select required columns
results = predictions.select("transaction_id", "prediction").withColumnRenamed("prediction", "fraud_probability")

# Write to feature table
results.write.saveAsTable("workspace.${MLPAB_DATABRICKS_SCHEMA}.ccprede0b071")

# Enable online serving
spark.sql(f"CREATE OR REFRESH LIVE TABLE workspace.${{MLPAB_DATABRICKS_SCHEMA}}.ccprede0b071 ONLINE STORE")

print("Scoring complete. Results written to ccprede0b071.")