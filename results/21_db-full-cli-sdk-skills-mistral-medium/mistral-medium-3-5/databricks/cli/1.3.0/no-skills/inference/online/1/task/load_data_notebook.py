# Databricks notebook source
# MAGIC %sql
# MAGIC -- Step 1: Create the Delta table from CSV
# MAGIC CREATE TABLE IF NOT EXISTS workspace.mlpab7c75f6.profilesd8bd1d
# MAGIC USING CSV OPTIONS (
# MAGIC   path '/Volumes/workspace/mlpab7c75f6/features_volume/features.csv',
# MAGIC   header 'true',
# MAGIC   inferSchema 'true'
# MAGIC );

# MAGIC %sql
# MAGIC -- Step 2: Verify the table was created
# MAGIC SELECT COUNT(*) FROM workspace.mlpab7c75f6.profilesd8bd1d;

# MAGIC %sql
# MAGIC -- Step 3: Query the table for lookup keys and write results to a temp table
# MAGIC -- First, read the lookup keys
# MAGIC CREATE OR REPLACE TEMP VIEW lookup_keys_view AS
# MAGIC SELECT trim(value) as account_id FROM json_tuple_keys(load_json('/Volumes/workspace/mlpab7c75f6/features_volume/lookup_keys.txt'));

# MAGIC %python
# MAGIC # Step 4: Query the table and write results
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
# MAGIC # Write results
# MAGIC with open('/Volumes/workspace/mlpab7c75f6/features_volume/answers.json', 'w') as f:
# MAGIC     json.dump({"vectors": vectors}, f)
