# Databricks notebook source
print("Testing simple read and write...")

volume_path = "/Volumes/workspace/mlpabfcf9c1/ccfraud_data/"
transactions_path = f"{volume_path}transactions.csv"

print(f"Reading from: {transactions_path}")

df = spark.read.csv(transactions_path, header=True, inferSchema=True)
print(f"Read {df.count()} rows")
print(f"Columns: {df.columns}")

# Try to write to a table
df.write.mode("overwrite").saveAsTable("workspace.mlpabfcf9c1.test_table")
print("Table created")
