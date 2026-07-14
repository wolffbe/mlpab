# Databricks notebook source
# MAGIC %sql
# MAGIC -- Step 1: Create the Delta table from CSV
# MAGIC CREATE TABLE IF NOT EXISTS workspace.mlpab7c75f6.profilesd8bd1d
# MAGIC USING CSV OPTIONS (
# MAGIC   path 'Volumes/workspace/mlpab7c75f6/features_volume/features.csv',
# MAGIC   header 'true',
# MAGIC   inferSchema 'true'
# MAGIC );

# MAGIC %sql
# MAGIC -- Step 2: Verify the table was created
# MAGIC SELECT COUNT(*) FROM workspace.mlpab7c75f6.profilesd8bd1d;
