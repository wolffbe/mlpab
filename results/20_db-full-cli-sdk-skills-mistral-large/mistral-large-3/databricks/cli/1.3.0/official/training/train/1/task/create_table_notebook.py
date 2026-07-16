# Databricks notebook source
# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS workspace.mlpaba53a68.predictionsac536a (
# MAGIC   row_id STRING,
# MAGIC   score DOUBLE
# MAGIC );
# MAGIC 
# MAGIC -- Load data from the predictions file in the workspace
# MAGIC COPY INTO workspace.mlpaba53a68.predictionsac536a
# MAGIC FROM "/Workspace/Users/benedict@logicalclocks.com/mlpaba53a68/predictions.csv"
# MAGIC FILEFORMAT = CSV
# MAGIC FORMAT_OPTIONS ('header' = 'true', 'inferSchema' = 'true');