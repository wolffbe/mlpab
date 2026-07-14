#!/usr/bin/env python3
"""
Build the training dataset for churn prediction.

Steps:
1. Create a volume in the schema to store input files.
2. Upload all input files to the volume.
3. For each (account_id, label_time) in labels.csv, join the most recent feature values
   from all source tables at or before label_time.
4. Write the result to a Delta table named `churntrainingaf8b21` with version 1.
"""

import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog
from databricks.sdk.service import files
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, max as spark_max
from pyspark.sql.window import Window

# Initialize Databricks SDK and Spark
w = WorkspaceClient()
spark = SparkSession.builder.getOrCreate()

# Environment variables
schema_name = os.getenv("MLPAB_DATABRICKS_SCHEMA")
prefix = os.getenv("MLPAB_DATABRICKS_PREFIX")
volume_name = f"{prefix}_input_data"
table_name = "churntrainingaf8b21"

# Create a volume to store input files
print(f"Creating volume {schema_name}.{volume_name}...")
w.volumes.create(
    catalog_name=schema_name.split(".")[0],
    schema_name=schema_name.split(".")[1],
    name=volume_name,
    volume_type=catalog.VolumeType.MANAGED,
)

# Upload input files to the volume
input_files = [
    "transactions.csv",
    "profiles.csv", 
    "activity.csv",
    "account_health.csv",
    "transactions_late.csv",
    "labels.csv",
]

volume_path = f"/Volumes/{schema_name}/{volume_name}"
for file_name in input_files:
    local_path = f"./data/{file_name}"
    remote_path = f"{volume_path}/{file_name}"
    print(f"Uploading {local_path} to {remote_path}...")
    w.files.upload(remote_path=remote_path, source_path=local_path)

# Read input files from the volume
print("Reading input files from volume...")
base_path = f"/Volumes/{schema_name}/{volume_name}"
transactions_df = spark.read.csv(f"{base_path}/transactions.csv", header=True, inferSchema=True)
profiles_df = spark.read.csv(f"{base_path}/profiles.csv", header=True, inferSchema=True)
activity_df = spark.read.csv(f"{base_path}/activity.csv", header=True, inferSchema=True)
health_df = spark.read.csv(f"{base_path}/account_health.csv", header=True, inferSchema=True)
transactions_late_df = spark.read.csv(f"{base_path}/transactions_late.csv", header=True, inferSchema=True)
labels_df = spark.read.csv(f"{base_path}/labels.csv", header=True, inferSchema=True)

# Union transactions and transactions_late
transactions_df = transactions_df.union(transactions_late_df)

# For each table, filter rows where event_time <= label_time and get the most recent row
print("Joining feature data...")

# Define a function to get the most recent row for each account_id at or before label_time
def get_most_recent(df, account_id_col, event_time_col, label_time_col):
    window = Window.partitionBy(account_id_col).orderBy(col(event_time_col).desc())
    filtered_df = df.filter(col(event_time_col) <= col(label_time_col))
    ranked_df = filtered_df.withColumn("rank", col("rank").over(window))
    return ranked_df.filter(col("rank") == 1).drop("rank")

# Join all features with labels
# Start with labels
result_df = labels_df

# Join transactions
transactions_window = Window.partitionBy("account_id").orderBy(col("event_time").desc())
transactions_ranked = transactions_df.withColumn("rank", col("rank").over(transactions_window))
transactions_latest = transactions_ranked.filter(col("rank") == 1).drop("rank", "event_time")
result_df = result_df.join(
    transactions_latest, 
    on="account_id", 
    how="left"
)

# Join profiles
profiles_window = Window.partitionBy("account_id").orderBy(col("event_time").desc())
profiles_ranked = profiles_df.withColumn("rank", col("rank").over(profiles_window))
profiles_latest = profiles_ranked.filter(col("rank") == 1).drop("rank", "event_time")
result_df = result_df.join(
    profiles_latest, 
    on="account_id", 
    how="left"
)

# Join activity
activity_window = Window.partitionBy("account_id").orderBy(col("event_time").desc())
activity_ranked = activity_df.withColumn("rank", col("rank").over(activity_window))
activity_latest = activity_ranked.filter(col("rank") == 1).drop("rank", "event_time")
result_df = result_df.join(
    activity_latest, 
    on="account_id", 
    how="left"
)

# Join health
health_window = Window.partitionBy("account_id").orderBy(col("event_time").desc())
health_ranked = health_df.withColumn("rank", col("rank").over(health_window))
health_latest = health_ranked.filter(col("rank") == 1).drop("rank", "event_time")
result_df = result_df.join(
    health_latest, 
    on="account_id", 
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
print(f"Writing result to {schema_name}.{table_name}...")
result_df.write.format("delta").mode("overwrite").saveAsTable(f"{schema_name}.{table_name}")

print("Training dataset created successfully!")