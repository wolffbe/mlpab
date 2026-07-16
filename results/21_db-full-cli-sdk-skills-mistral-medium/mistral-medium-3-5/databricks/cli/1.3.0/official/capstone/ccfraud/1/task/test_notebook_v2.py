# Databricks notebook source
print("Hello from notebook")

# Try to read CSV using dbfs path
df = spark.read.csv("dbfs:/user/benedict@hopsworks.ai/mlpabfcf9c1/data/transactions.csv", header=True, inferSchema=True)
print(f"Read {df.count()} rows")
print(f"Columns: {df.columns}")
