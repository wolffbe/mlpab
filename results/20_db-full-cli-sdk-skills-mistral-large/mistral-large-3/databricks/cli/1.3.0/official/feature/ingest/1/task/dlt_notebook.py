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
(delta_table_path = f"{spark.conf.get('spark.databricks.delta.target')}.transactions4adadd")
df.write.format("delta").mode("overwrite").saveAsTable("transactions4adadd")

# Set the record key and event-time column properties
spark.sql(f"""
ALTER TABLE {spark.conf.get('spark.databricks.delta.target')}.transactions4adadd
SET TBLPROPERTIES (
  'delta.feature.recordKey' = 'row_id',
  'delta.feature.eventTimeColumn' = 'event_time'
)
""")