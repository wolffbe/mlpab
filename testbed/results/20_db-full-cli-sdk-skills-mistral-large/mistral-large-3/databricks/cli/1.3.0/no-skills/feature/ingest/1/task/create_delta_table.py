# Databricks notebook source
# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS workspace.mlpabd62957.transactions4adadd (
# MAGIC     row_id STRING,
# MAGIC     account_id STRING,
# MAGIC     event_time BIGINT,
# MAGIC     amount DOUBLE,
# MAGIC     category STRING
# MAGIC ) USING DELTA;

# COMMAND ----------

# Load data from the first export
spark.read.csv("dbfs:/Volumes/workspace/mlpabd62957/ingest_volume/transactions_export_1.csv", header=True, inferSchema=True).write.mode("append").saveAsTable("workspace.mlpabd62957.transactions4adadd")

# Load data from the second export
spark.read.csv("dbfs:/Volumes/workspace/mlpabd62957/ingest_volume/transactions_export_2.csv", header=True, inferSchema=True).write.mode("append").saveAsTable("workspace.mlpabd62957.transactions4adadd")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Register the table as a feature table
# MAGIC CREATE FEATURE TABLE IF NOT EXISTS workspace.mlpabd62957.transactions4adadd
# MAGIC AS SELECT * FROM workspace.mlpabd62957.transactions4adadd;
# MAGIC 
# MAGIC -- Enable online access for low-latency lookup
# MAGIC CREATE OR REFRESH ONLINE TABLE workspace.mlpabd62957.transactions4adadd
# MAGIC FROM FEATURES workspace.mlpabd62957.transactions4adadd;