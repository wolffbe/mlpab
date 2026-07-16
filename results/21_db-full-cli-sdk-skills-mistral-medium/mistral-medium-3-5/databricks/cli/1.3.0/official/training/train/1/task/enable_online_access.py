# Databricks notebook source
# MAGIC %sql
# MAGIC -- Try to enable online access for the table
# MAGIC -- First, let's check if we can create a feature table
# MAGIC SHOW TABLES IN workspace.mlpab155832;
# MAGIC 
# MAGIC -- Check the current table
# MAGIC DESCRIBE TABLE workspace.mlpab155832.predictionsd631cc;
# MAGIC 
# MAGIC -- Try to create a feature table (if this syntax is supported)
# MAGIC CREATE FEATURE TABLE IF NOT EXISTS workspace.mlpab155832.predictionsd631cc_v1 
# MAGIC USING DELTA 
# MAGIC AS SELECT * FROM workspace.mlpab155832.predictionsd631cc;
# MAGIC 
# MAGIC -- If that doesn't work, try to alter the existing table
# MAGIC ALTER TABLE workspace.mlpab155832.predictionsd631cc 
# MAGIC SET TBLPROPERTIES ('feature_table' = 'true', 'record_key' = 'row_id');
# MAGIC 
# MAGIC -- Check if the table has the right properties
# MAGIC SHOW TBLPROPERTIES workspace.mlpab155832.predictionsd631cc;