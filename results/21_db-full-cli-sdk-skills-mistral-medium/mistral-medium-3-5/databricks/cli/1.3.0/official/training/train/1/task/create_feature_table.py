# Databricks notebook source
# MAGIC %sql
# MAGIC -- Create a feature table from the existing predictions table
# MAGIC CREATE OR REPLACE TABLE workspace.mlpab155832.predictionsd631cc_v1 
# MAGIC AS SELECT * FROM workspace.mlpab155832.predictionsd631cc;
# MAGIC 
# MAGIC -- Add primary key constraint
# MAGIC ALTER TABLE workspace.mlpab155832.predictionsd631cc_v1 
# MAGIC ADD CONSTRAINT pk_predictionsd631cc_v1 PRIMARY KEY (row_id);
# MAGIC 
# MAGIC -- Verify the table
# MAGIC SHOW TABLES IN workspace.mlpab155832;
# MAGIC DESCRIBE TABLE workspace.mlpab155832.predictionsd631cc_v1;