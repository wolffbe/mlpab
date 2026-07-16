# Databricks notebook source
# MAGIC %md
# MAGIC ### Ingest Transactions Data
# MAGIC 
# MAGIC This notebook ingests the transactions data from CSV files into a Delta table, deduplicating by `row_id`.

# COMMAND ----------

from pyspark.sql import functions as F

# Paths to the CSV files
csv_path_1 = "/dbfs/tmp/mlpaba52a53/transactions_export_1.csv"
csv_path_2 = "/dbfs/tmp/mlpaba52a53/transactions_export_2.csv"

# Read the CSV files
schema = "row_id STRING, account_id STRING, event_time BIGINT, amount DOUBLE, category STRING"
df1 = spark.read.csv(csv_path_1, header=True, schema=schema)
df2 = spark.read.csv(csv_path_2, header=True, schema=schema)

# Union and deduplicate by row_id
df = df1.union(df2).dropDuplicates(["row_id"])

# Write to Delta table
# Write to Delta table
df.write.format("delta").mode("overwrite").saveAsTable("transactions4adadd")
df.write.format("delta").mode("overwrite").saveAsTable("transactions4adadd")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register for Online Access
# MAGIC 
# MAGIC The table is now available for online/real-time access.