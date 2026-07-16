# Databricks notebook source
# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS workspace.${MLPAB_DATABRICKS_PREFIX}.predictionsac536a (
# MAGIC   row_id STRING,
# MAGIC   score DOUBLE
# MAGIC );
# MAGIC 
# MAGIC -- Load data from the predictions file in the workspace
# MAGIC COPY INTO workspace.${MLPAB_DATABRICKS_PREFIX}.predictionsac536a
# MAGIC FROM "/Workspace/Users/benedict@logicalclocks.com/${MLPAB_DATABRICKS_PREFIX}/predictions.csv"
# MAGIC FILEFORMAT = CSV
# MAGIC FORMAT_OPTIONS ('header' = 'true', 'inferSchema' = 'true')
# MAGIC;