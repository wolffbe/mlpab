# Databricks notebook source
print("Testing volume read...")

# Try different path formats
paths = [
    "/Volumes/workspace/mlpabfcf9c1/ccfraud_data/transactions.csv",
    "dbfs:/Volumes/workspace/mlpabfcf9c1/ccfraud_data/transactions.csv",
    "/dbfs/Volumes/workspace/mlpabfcf9c1/ccfraud_data/transactions.csv",
]

for path in paths:
    try:
        df = spark.read.csv(path, header=True, inferSchema=True)
        print(f"SUCCESS with path: {path}, count: {df.count()}")
        break
    except Exception as e:
        print(f"FAILED with path: {path}, error: {e}")
