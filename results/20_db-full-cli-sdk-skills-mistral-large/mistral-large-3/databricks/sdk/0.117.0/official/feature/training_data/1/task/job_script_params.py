#!/usr/bin/env python3
"""
PySpark script to build the training dataset for churn prediction.

Steps:
1. Read input files from the volume.
2. For each (account_id, label_time) in labels.csv, join the most recent feature values
   from all source tables at or before label_time.
3. Write the result to a Delta table named `churntrainingaf8b21`.
"""

import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.window import Window

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--schema", type=str, required=True, help="Schema name")
parser.add_argument("--volume", type=str, required=True, help="Volume name")
args = parser.parse_args()

# Initialize Spark
spark = SparkSession.builder.getOrCreate()

# Read input files from the volume
base_path = f"/Volumes/{args.schema.split('.')[0]}/{args.schema.split('.')[1]}/{args.volume}"
transactions_df = spark.read.csv(f"{base_path}/transactions.csv", header=True, inferSchema=True)
profiles_df = spark.read.csv(f"{base_path}/profiles.csv", header=True, inferSchema=True)
activity_df = spark.read.csv(f"{base_path}/activity.csv", header=True, inferSchema=True)
health_df = spark.read.csv(f"{base_path}/account_health.csv", header=True, inferSchema=True)
transactions_late_df = spark.read.csv(f"{base_path}/transactions_late.csv", header=True, inferSchema=True)
labels_df = spark.read.csv(f"{base_path}/labels.csv", header=True, inferSchema=True)

# Union transactions and transactions_late
transactions_df = transactions_df.union(transactions_late_df)

# For each (account_id, label_time), get the most recent feature values at or before label_time
print("Joining feature data...")

# Start with labels
result_df = labels_df

# Join transactions: most recent at or before label_time
transactions_window = Window.partitionBy("account_id", "label_time").orderBy(col("event_time").desc())
transactions_filtered = transactions_df.filter(col("event_time") <= col("label_time"))
transactions_ranked = transactions_filtered.withColumn("rank", col("rank").over(transactions_window))
transactions_latest = transactions_ranked.filter(col("rank") == 1).drop("rank", "event_time")
result_df = result_df.join(
    transactions_latest, 
    on=["account_id", "label_time"], 
    how="left"
)

# Join profiles: most recent at or before label_time
profiles_window = Window.partitionBy("account_id", "label_time").orderBy(col("event_time").desc())
profiles_filtered = profiles_df.filter(col("event_time") <= col("label_time"))
profiles_ranked = profiles_filtered.withColumn("rank", col("rank").over(profiles_window))
profiles_latest = profiles_ranked.filter(col("rank") == 1).drop("rank", "event_time")
result_df = result_df.join(
    profiles_latest, 
    on=["account_id", "label_time"], 
    how="left"
)

# Join activity: most recent at or before label_time
activity_window = Window.partitionBy("account_id", "label_time").orderBy(col("event_time").desc())
activity_filtered = activity_df.filter(col("event_time") <= col("label_time"))
activity_ranked = activity_filtered.withColumn("rank", col("rank").over(activity_window))
activity_latest = activity_ranked.filter(col("rank") == 1).drop("rank", "event_time")
result_df = result_df.join(
    activity_latest, 
    on=["account_id", "label_time"], 
    how="left"
)

# Join health: most recent at or before label_time
health_window = Window.partitionBy("account_id", "label_time").orderBy(col("event_time").desc())
health_filtered = health_df.filter(col("event_time") <= col("label_time"))
health_ranked = health_filtered.withColumn("rank", col("rank").over(health_window))
health_latest = health_ranked.filter(col("rank") == 1).drop("rank", "event_time")
result_df = result_df.join(
    health_latest, 
    on=["account_id", "label_time"], 
    how="left"
)

# Select the required columns
result_df = result_df.select(
    "account_id", 
    "label_time", 
    "amount", 
    "balance", 
    "credit_score", 
    "tier", 
    "sessions_7d", 
    "health_score", 
    "churned"
)

# Write the result to a Delta table
print(f"Writing result to {args.schema}.churntrainingaf8b21...")
result_df.write.format("delta").mode("overwrite").saveAsTable(f"{args.schema}.churntrainingaf8b21")

print("Training dataset created successfully!")