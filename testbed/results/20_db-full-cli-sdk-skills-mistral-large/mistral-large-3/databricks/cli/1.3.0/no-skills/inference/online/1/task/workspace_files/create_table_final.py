# Databricks notebook source
# MAGIC %sql
CREATE TABLE IF NOT EXISTS ${MLPAB_DATABRICKS_SCHEMA}.profiles395e7c (account_id STRING, f1 DOUBLE, f2 DOUBLE, f3 DOUBLE, f4 DOUBLE);

# COMMAND ----------

# MAGIC %sql
INSERT INTO ${MLPAB_DATABRICKS_SCHEMA}.profiles395e7c
SELECT * FROM csv."/Users/benedict@logicalclocks.com/${MLPAB_DATABRICKS_PREFIX}/features.csv" (header => true, inferSchema => true);