# Databricks notebook source
# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS workspace.mlpab57faf4.customersc23945 (
# MAGIC   row_id STRING,
# MAGIC   name STRING,
# MAGIC   balance_eur DOUBLE,
# MAGIC   updated_at BIGINT
# MAGIC ) USING DELTA
# MAGIC PARTITIONED BY (updated_at)
# MAGIC LOCATION '/tmp/'${MLPAB_DATABRICKS_PREFIX}'_customers/customersc23945_v1';
# MAGIC 
# MAGIC -- Load data
# MAGIC COPY INTO workspace.mlpab57faf4.customersc23945
# MAGIC FROM '/tmp/'${MLPAB_DATABRICKS_PREFIX}'_customers/initial_export.csv'
# MAGIC FILEFORMAT = CSV
# MAGIC FORMAT_OPTIONS ('header' = 'true', 'inferSchema' = 'false')
# MAGIC PATTERN = '*.csv';