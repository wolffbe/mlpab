# Databricks notebook source
# MAGIC %python
# MAGIC import json
# MAGIC 
# MAGIC # Read lookup keys
# MAGIC with open('/Volumes/workspace/mlpab7c75f6/features_volume/lookup_keys.txt', 'r') as f:
# MAGIC     lookup_keys = [line.strip() for line in f.readlines()]
# MAGIC 
# MAGIC # Query the Delta table for each key
# MAGIC vectors = {}
# MAGIC for key in lookup_keys:
# MAGIC     df = spark.sql(f"SELECT f1, f2, f3, f4 FROM workspace.mlpab7c75f6.profilesd8bd1d WHERE account_id = '{key}'")
# MAGIC     if df.count() > 0:
# MAGIC         row = df.collect()[0]
# MAGIC         vectors[key] = [float(row.f1), float(row.f2), float(row.f3), float(row.f4)]
# MAGIC 
# MAGIC # Write results to volume
# MAGIC with open('/Volumes/workspace/mlpab7c75f6/features_volume/answers.json', 'w') as f:
# MAGIC     json.dump({"vectors": vectors}, f)
