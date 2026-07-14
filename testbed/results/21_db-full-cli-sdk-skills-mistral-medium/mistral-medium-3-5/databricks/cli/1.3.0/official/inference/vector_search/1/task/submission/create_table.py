# Databricks notebook source
# MAGIC %python
# MAGIC 
# MAGIC # Read the CSV and create a Delta table
# MAGIC import pandas as pd
# MAGIC import ast
# MAGIC 
# MAGIC # Read items.csv
# MAGIC df = pd.read_csv('/Workspace/Users/benedict@hopsworks.ai/mlpabb9b3c4/data/items.csv')
# MAGIC 
# MAGIC # Convert embedding string to array
# MAGIC df['embedding'] = df['embedding'].apply(lambda x: ast.literal_eval(x))
# MAGIC 
# MAGIC # Write to Delta table
# MAGIC spark.createDataFrame(df).write.mode('overwrite').saveAsTable('workspace.mlpabb9b3c4.itemsffc8a7')