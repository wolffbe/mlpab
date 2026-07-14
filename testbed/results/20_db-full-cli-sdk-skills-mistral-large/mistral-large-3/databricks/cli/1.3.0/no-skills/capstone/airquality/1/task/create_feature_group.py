# Databricks notebook source
# Read the CSV file from the volume
df = spark.read.csv("/Volumes/workspace/mlpab6e9823/data/airquality_history.csv", header=True, inferSchema=True)

# Feature engineering
from pyspark.sql.functions import lag, mean
from pyspark.sql.window import Window

window_spec = Window.orderBy("date")

# Lag features
for lag_days in [1, 2, 3, 7]:
    df = df.withColumn(f"pm25_lag{lag_days}", lag("pm25", lag_days).over(window_spec))

# Rolling statistics
rolling_window = Window.orderBy("date").rowsBetween(-6, 0)
df = df.withColumn("pm25_rolling_mean_7d", mean("pm25").over(rolling_window))

# Write to feature group
df.write.saveAsTable("workspace.mlpab6e9823.airqfcd91b")