# Databricks notebook source
# MAGIC %python

# COMMAND ----------

# Read the CSV file from DBFS
import pandas as pd

# Define valid categories
valid_categories = {'grocery', 'travel', 'salary', 'rent', 'other'}

# Read the CSV
file_path = '/dbfs/FileStore/tmp/valid_events.csv'
df = pd.read_csv(file_path)

# Filter valid rows (already filtered, but let's verify)
print(f"Total rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(df.head())
