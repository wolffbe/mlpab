# Databricks notebook source
# MAGIC %python
# MAGIC # Read CSV from volume
# MAGIC df = spark.read.csv('/Volumes/workspace/mlpab7c75f6/features_volume/features.csv', header=True, inferSchema=True)
# MAGIC 
# MAGIC # Save as Delta table
# MAGIC df.write.saveAsTable('workspace.mlpab7c75f6.profilesd8bd1d')
# MAGIC 
# MAGIC # Verify
# MAGIC display(spark.sql('SELECT COUNT(*) FROM workspace.mlpab7c75f6.profilesd8bd1d'))
