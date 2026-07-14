# Databricks notebook source
# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS workspace.mlpab8a9a8e.rawa9a7eb8 (
# MAGIC   row_id STRING,
# MAGIC   a_val DOUBLE
# MAGIC ) USING DELTA;
# MAGIC 
# MAGIC COPY INTO workspace.mlpab8a9a8e.rawa9a7eb8
# MAGIC FROM (SELECT * FROM csv.`dbfs:/Volumes/workspace/mlpab8a9a8e/staging/raw_a.csv`)
# MAGIC FILEFORMAT = CSV
# MAGIC FORMAT_OPTIONS ('header' = 'true', 'inferSchema' = 'true')
# MAGIC PATTERN = "*"
# MAGIC;
# MAGIC 
# MAGIC CREATE TABLE IF NOT EXISTS workspace.mlpab8a9a8e.rawb9a7eb8 (
# MAGIC   row_id STRING,
# MAGIC   b_val DOUBLE
# MAGIC ) USING DELTA;
# MAGIC 
# MAGIC COPY INTO workspace.mlpab8a9a8e.rawb9a7eb8
# MAGIC FROM (SELECT * FROM csv.`dbfs:/Volumes/workspace/mlpab8a9a8e/staging/raw_b.csv`)
# MAGIC FILEFORMAT = CSV
# MAGIC FORMAT_OPTIONS ('header' = 'true', 'inferSchema' = 'true')
# MAGIC PATTERN = "*"
# MAGIC;
# MAGIC 
# MAGIC -- Create derived table
# MAGIC CREATE TABLE IF NOT EXISTS workspace.mlpab8a9a8e.derived9a7eb8 AS
# MAGIC SELECT 
# MAGIC   a.row_id,
# MAGIC   ROUND(a.a_val + b.b_val, 6) AS col_sum
# MAGIC FROM workspace.mlpab8a9a8e.rawa9a7eb8 a
# MAGIC JOIN workspace.mlpab8a9a8e.rawb9a7eb8 b
# MAGIC ON a.row_id = b.row_id;
# MAGIC 
# MAGIC -- Enable online table for derived9a7eb8
# MAGIC CREATE OR REFRESH LIVE TABLE workspace.mlpab8a9a8e.derived9a7eb8_rt
# MAGIC AS SELECT * FROM workspace.mlpab8a9a8e.derived9a7eb8;