# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# COMMAND ----------
# Read source data from UC Volume
txn = spark.read.csv(
    "/Volumes/workspace/mlpab9e4ddc/data_files/transactions.csv",
    header=True, inferSchema=True
)
fx = spark.read.csv(
    "/Volumes/workspace/mlpab9e4ddc/data_files/fx_rates.csv",
    header=True, inferSchema=True
)

# COMMAND ----------
# Join to get fx_rate
txn = txn.join(fx, on="currency", how="left")

# amount_usd = amount * fx_rate
txn = txn.withColumn("amount_usd", F.col("amount") * F.col("fx_rate"))

# is_weekend: event_time is epoch milliseconds; dayofweek: 1=Sunday, 7=Saturday
txn = txn.withColumn(
    "is_weekend",
    F.when(
        F.dayofweek(F.from_unixtime(F.col("event_time") / 1000)).isin([1, 7]),
        1
    ).otherwise(0)
)

# amount_7d: rolling 7-day sum of amount per account_id
# Window: [event_time - 7 days, event_time] inclusive, in milliseconds
seven_days_ms = 7 * 24 * 60 * 60 * 1000
w = (
    Window.partitionBy("account_id")
    .orderBy(F.col("event_time").cast("long"))
    .rangeBetween(-seven_days_ms, 0)
)
txn = txn.withColumn("amount_7d", F.sum("amount").over(w))

# COMMAND ----------
# Select final columns
result = txn.select(
    "row_id",
    "account_id",
    F.col("event_time").cast("long").alias("event_time"),
    F.col("amount_usd").cast("double").alias("amount_usd"),
    F.col("is_weekend").cast("int").alias("is_weekend"),
    F.col("amount_7d").cast("double").alias("amount_7d")
)

# Write as Delta table in UC schema
result.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("workspace.mlpab9e4ddc.featuresb1ea93")

print("Feature table created successfully")
spark.sql("SELECT * FROM workspace.mlpab9e4ddc.featuresb1ea93 LIMIT 5").show()

# COMMAND ----------
# Enable Change Data Feed for online table support
spark.sql("""
ALTER TABLE workspace.mlpab9e4ddc.featuresb1ea93
SET TBLPROPERTIES (
    'delta.enableChangeDataFeed' = 'true'
)
""")

print("Done!")
