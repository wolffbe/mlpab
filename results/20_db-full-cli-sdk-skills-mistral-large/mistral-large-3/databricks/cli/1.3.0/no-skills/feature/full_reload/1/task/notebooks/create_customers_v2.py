# Databricks notebook source
# MAGIC %sql
# MAGIC -- Drop the old table if it exists
# MAGIC DROP TABLE IF EXISTS workspace.mlpab57faf4.customersc23945;
# MAGIC 
# MAGIC -- Create version 2 of the table with the new schema
# MAGIC CREATE TABLE workspace.mlpab57faf4.customersc23945 (
# MAGIC   row_id STRING,
# MAGIC   full_name STRING,
# MAGIC   balance DOUBLE,
# MAGIC   currency STRING,
# MAGIC   updated_at BIGINT
# MAGIC ) USING DELTA
# MAGIC PARTITIONED BY (updated_at)
# MAGIC LOCATION '/tmp/'${MLPAB_DATABRICKS_PREFIX}'_customers/customersc23945_v2';
# MAGIC 
# MAGIC -- Load data from the new export
# MAGIC COPY INTO workspace.mlpab57faf4.customersc23945
# MAGIC FROM '/tmp/'${MLPAB_DATABRICKS_PREFIX}'_customers/new_export.csv'
# MAGIC FILEFORMAT = CSV
# MAGIC FORMAT_OPTIONS ('header' = 'true', 'inferSchema' = 'false')
# MAGIC PATTERN = '*.csv';