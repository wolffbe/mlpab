# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Read raw data from Volume
transactions_path = "/Volumes/workspace/mlpab858a9e/raw/transactions.csv"
fx_rates_path = "/Volumes/workspace/mlpab858a9e/raw/fx_rates.csv"

transactions = spark.read.format("csv").option("header", "true").load(transactions_path)
fx_rates = spark.read.format("csv").option("header", "true").load(fx_rates_path)

# Join transactions with fx_rates to compute amount_usd
joined = transactions.join(fx_rates, "currency", "left") \
    .withColumn("event_time", (F.col("event_time") / 1000).cast("timestamp")) \
    .withColumn("amount_usd", F.col("amount") * F.col("fx_rate"))

# Add is_weekend column
joined = joined.withColumn(
    "is_weekend",
    F.when(F.dayofweek(F.col("event_time")).isin([1, 7]), 1).otherwise(0)
)

# Compute amount_7d: sum of amount for the account over the last 7 days (inclusive)
window = Window.partitionBy("account_id").orderBy("event_time").rowsBetween(-6, 0)
features = joined.withColumn("amount_7d", F.sum("amount").over(window))

# Write to feature table
features.select(
    "row_id",
    "account_id",
    F.col("event_time").cast("long") * 1000,  # Convert back to epoch milliseconds
    "amount_usd",
    "is_weekend",
    "amount_7d"
).write.saveAsTable("workspace.mlpab858a9e.featuresfbc05f")