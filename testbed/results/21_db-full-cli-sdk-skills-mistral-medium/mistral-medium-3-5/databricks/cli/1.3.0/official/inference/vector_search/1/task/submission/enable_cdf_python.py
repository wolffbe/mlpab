# Databricks notebook source
# MAGIC %python
# MAGIC 
# MAGIC # Enable change data feed on the table
# MAGIC spark.sql("ALTER TABLE workspace.mlpabb9b3c4.itemsffc8a7 SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
# MAGIC 
# MAGIC print("CDF enabled successfully")
