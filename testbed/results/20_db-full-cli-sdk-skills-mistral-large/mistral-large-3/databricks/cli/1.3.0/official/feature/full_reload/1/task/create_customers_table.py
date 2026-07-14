# Databricks notebook source
# MAGIC %sql
CREATE TABLE IF NOT EXISTS workspace.${MLPAB_DATABRICKS_SCHEMA}.customersc23945_v1 (
  row_id STRING,
  name STRING,
  balance_eur DOUBLE,
  updated_at BIGINT
)
USING DELTA
LOCATION 'dbfs:/Volumes/workspace/mlpab7d1728/uploads/customersc23945_v1'
AS SELECT * FROM csv."`dbfs:/Volumes/workspace/mlpab7d1728/uploads/initial_export.csv`" (header => true, inferSchema => true);

-- Register as feature table
CREATE FEATURE TABLE IF NOT EXISTS workspace.${MLPAB_DATABRICKS_SCHEMA}.customersc23945
  FROM workspace.${MLPAB_DATABRICKS_SCHEMA}.customersc23945_v1
  KEYS (row_id)
  TIMESTAMP KEY (updated_at);