# Databricks notebook source
import dlt
from pyspark.sql.functions import col, lag, mean, datediff, to_date
from pyspark.sql.window import Window

# Read the raw data
raw_data_path = "/Volumes/{catalog}/{schema}/data/airquality_history.csv"
history_df = spark.read.csv(raw_data_path, header=True, inferSchema=True)

# Feature engineering
window_spec = Window.orderBy("date")

# Lag features
for lag_days in [1, 2, 3, 7]:
    history_df = history_df.withColumn(f"pm25_lag{lag_days}", lag("pm25", lag_days).over(window_spec))

# Rolling statistics
rolling_window = Window.orderBy("date").rowsBetween(-6, 0)
history_df = history_df.withColumn("pm25_rolling_mean_7d", mean("pm25").over(rolling_window))

# Write to feature group
@dlt.table(name="airqfcd91b")
def create_feature_group():
    return history_df