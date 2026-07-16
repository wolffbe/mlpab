# Databricks notebook source
print("Testing volume read...")

# Try the path that should work
path = "/Volumes/workspace/mlpabfcf9c1/ccfraud_data/transactions.csv"

try:
    df = spark.read.csv(path, header=True, inferSchema=True)
    print(f"SUCCESS with path: {path}")
    print(f"Count: {df.count()}")
    print(f"Columns: {df.columns}")
    print(f"First row: {df.first()}")
except Exception as e:
    print(f"FAILED with path: {path}, error: {e}")
    import traceback
    traceback.print_exc()
