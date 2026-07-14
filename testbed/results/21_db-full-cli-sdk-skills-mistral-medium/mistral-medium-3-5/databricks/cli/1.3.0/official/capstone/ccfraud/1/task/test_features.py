# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.sql.window import Window

print("Testing feature engineering...")

volume_path = "/Volumes/workspace/mlpabfcf9c1/ccfraud_data/"
transactions_path = f"{volume_path}transactions.csv"

df = spark.read.csv(transactions_path, header=True, inferSchema=True)
df = df.withColumn("datetime", F.to_timestamp("datetime", "yyyy-MM-dd'T'HH:mm:ss'Z'"))

print(f"Initial count: {df.count()}")

# Add simple features
df = df.withColumn("hour_of_day", F.hour("datetime"))
df = df.withColumn("amount_log", F.log1p("amount"))

print(f"After simple features: {df.count()}")

# Add window features
from pyspark.sql.window import Window
window_spec = Window.partitionBy("cc_num")
df = df.withColumn("amount_mean", F.mean("amount").over(window_spec))

print(f"After window features: {df.count()}")

# Write to table
df.write.mode("overwrite").saveAsTable("workspace.mlpabfcf9c1.test_features")
print("Feature table created")
