# Databricks notebook source
# MAGIC %md
# MAGIC ## Ingest Transactions Data
# MAGIC 
# MAGIC This notebook reads CSV files, deduplicates by row_id, and creates a feature table.

# COMMAND ----------

# MAGIC %python
# MAGIC import pandas as pd
# MAGIC from pyspark.sql import SparkSession
# MAGIC 
# MAGIC spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# MAGIC %python
# MAGIC # Read CSV files from workspace
# MAGIC # Note: Files need to be uploaded to workspace first
# MAGIC 
# MAGIC # For now, let's create the data inline
# MAGIC # We'll read the CSV files from the local data directory
# MAGIC # But since we're on the platform, we need to get the files there first
# MAGIC 
# MAGIC # Actually, let's use dbutils to read from workspace
# MAGIC dbutils.ls("/Users/benedict@logicalclocks.com/mlpabf8bf6d")

# COMMAND ----------

# MAGIC %python
# MAGIC # Read both CSV files
# MAGIC df1 = spark.read.csv("/Workspace/Users/benedict@logicalclocks.com/mlpabf8bf6d/transactions_export_1.csv", header=True, inferSchema=True)
# MAGIC df2 = spark.read.csv("/Workspace/Users/benedict@logicalclocks.com/mlpabf8bf6d/transactions_export_2.csv", header=True, inferSchema=True)
# MAGIC 
# MAGIC # Union and deduplicate
# MAGIC from pyspark.sql.window import Window
# MAGIC from pyspark.sql.functions import row_number
# MAGIC 
# MAGIC combined = df1.union(df2)
# MAGIC window = Window.partitionBy("row_id").orderBy(combined["event_time"].desc())
# MAGIC deduped = combined.withColumn("rn", row_number().over(window)).filter("rn = 1").drop("rn")
# MAGIC 
# MAGIC # Write to Unity Catalog table
# MAGIC deduped.write.format("delta").saveAsTable("workspace.mlpabf8bf6d.transactions30b87b")
