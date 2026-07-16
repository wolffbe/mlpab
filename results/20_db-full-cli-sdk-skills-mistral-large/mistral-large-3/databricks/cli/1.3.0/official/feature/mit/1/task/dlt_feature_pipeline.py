# Databricks notebook source
import dlt
from pyspark.sql.functions import col, expr, sum, when
from pyspark.sql.window import Window

# Read raw data from Volume
transactions_path = "/Volumes/workspace/mlpab858a9e/raw/transactions.csv"
fx_rates_path = "/Volumes/workspace/mlpab858a9e/raw/fx_rates.csv"

transactions = spark.read.format("csv").option("header", "true").load(transactions_path)
fx_rates = spark.read.format("csv").option("header", "true").load(fx_rates_path)

# Join transactions with fx_rates to compute amount_usd
joined = transactions.join(fx_rates, "currency", "left") \
    .withColumn("event_time", (col("event_time") / 1000).cast("timestamp")) \
    .withColumn("amount_usd", col("amount") * col("fx_rate"))

# Add is_weekend column
joined = joined.withColumn(
    "is_weekend",
    when(expr("dayofweek(event_time) = 1 OR dayofweek(event_time) = 7"), 1).otherwise(0)
)

# Compute amount_7d: sum of amount for the account over the last 7 days (inclusive)
window = Window.partitionBy("account_id").orderBy("event_time").rowsBetween(-6, 0)
features = joined.withColumn("amount_7d", sum("amount").over(window))

# Write to feature table
@dlt.table(
    name="featuresfbc05f",
    comment="Feature table for transactions with amount_usd, is_weekend, and amount_7d"
)
def create_features_table():
    return features.select(
        "row_id",
        "account_id",
        "event_time",
        "amount_usd",
        "is_weekend",
        "amount_7d"
    )