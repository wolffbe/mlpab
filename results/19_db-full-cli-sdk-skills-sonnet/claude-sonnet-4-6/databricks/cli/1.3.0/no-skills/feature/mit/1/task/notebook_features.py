# Databricks notebook source
# MAGIC %python

# COMMAND ----------
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Read input data
transactions = spark.read.option("header", "true").option("inferSchema", "true") \
    .csv("/Volumes/workspace/mlpabf1452c/mlpabf1452c_data/transactions.csv")

fx_rates = spark.read.option("header", "true").option("inferSchema", "true") \
    .csv("/Volumes/workspace/mlpabf1452c/mlpabf1452c_data/fx_rates.csv")

# Join with FX rates to get amount_usd
df = transactions.join(fx_rates, on="currency", how="left")
df = df.withColumn("amount_usd", F.col("amount") * F.col("fx_rate"))

# is_weekend: event_time is epoch milliseconds
df = df.withColumn(
    "event_time_ts",
    (F.col("event_time") / 1000).cast("timestamp")
)
df = df.withColumn(
    "day_of_week",
    F.dayofweek(F.to_utc_timestamp(F.col("event_time_ts"), "UTC"))
)
# dayofweek: 1=Sunday, 2=Monday, ..., 7=Saturday
df = df.withColumn(
    "is_weekend",
    F.when(F.col("day_of_week").isin(1, 7), 1).otherwise(0)
)

# amount_7d: sum of amount for same account over [event_time - 7 days, event_time] inclusive
# 7 days = 7 * 24 * 60 * 60 * 1000 milliseconds
seven_days_ms = 7 * 24 * 60 * 60 * 1000

window_7d = Window.partitionBy("account_id") \
    .orderBy("event_time") \
    .rangeBetween(-seven_days_ms, 0)

df = df.withColumn("amount_7d", F.sum("amount").over(window_7d))

# Select final columns
result = df.select(
    F.col("row_id"),
    F.col("account_id"),
    F.col("event_time"),
    F.col("amount_usd"),
    F.col("is_weekend"),
    F.col("amount_7d")
)

# Write as Delta table to Unity Catalog
result.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("workspace.mlpabf1452c.featuresb1ea93")

print("Feature table created successfully.")
result.show(5)

# COMMAND ----------
# Verify row count
count = spark.table("workspace.mlpabf1452c.featuresb1ea93").count()
print(f"Row count: {count}")
