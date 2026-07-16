# Databricks notebook source
print("Hello from notebook")
print(f"Schema: {os.getenv('MLPAB_DATABRICKS_SCHEMA', 'not set')}")

# Try to read CSV
df = spark.read.csv("/Workspace/Users/benedict@hopsworks.ai/mlpabfcf9c1/data/transactions.csv", header=True, inferSchema=True)
print(f"Read {df.count()} rows")
print(f"Columns: {df.columns}")
