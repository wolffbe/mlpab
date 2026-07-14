# Databricks notebook source
SCHEMA = "workspace.mlpab894195"
VOLUME_PATH = "/Volumes/workspace/mlpab894195/data_vol"

# COMMAND ----------
# Export derived table to CSV in the volume
spark.table(f"{SCHEMA}.derivedd05474").orderBy("row_id").coalesce(1).write.mode("overwrite").option("header", "true").csv(f"{VOLUME_PATH}/export_derived")

# List the file
import os
files = dbutils.fs.ls(f"{VOLUME_PATH}/export_derived/")
print("Exported files:", [f.name for f in files])

# Find the CSV file
csv_file = [f.path for f in files if f.name.endswith('.csv')][0]
print("CSV file:", csv_file)

# Read and print the content
content = spark.read.text(csv_file)
for row in content.collect():
    print(row['value'])
