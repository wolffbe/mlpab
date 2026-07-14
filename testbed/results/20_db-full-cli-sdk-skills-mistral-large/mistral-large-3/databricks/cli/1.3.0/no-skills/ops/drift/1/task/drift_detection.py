#!/usr/bin/env python3

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, mean, stddev, count, lag, abs
from pyspark.sql.window import Window
import json

# Initialize Spark
spark = SparkSession.builder.getOrCreate()

# Read the data from the Volume
df = spark.read.csv("dbfs:/Volumes/workspace/mlpabf81e30/mlpabf81e30_volume/features.csv", header=True, inferSchema=True)

# Extract date from event_time
df = df.withColumn("event_date", to_date(col("event_time")))

# Compute daily statistics for each feature
features = ["f1", "f2", "f3", "f4", "f5", "f6"]
stats_df = df.groupBy("event_date").agg(
    *[mean(col(f)).alias(f"{f}_mean") for f in features],
    *[stddev(col(f)).alias(f"{f}_stddev") for f in features],
    count("*").alias("count")
).orderBy("event_date")

# Define a window for rolling statistics
window = Window.orderBy("event_date")

# Compute rolling Z-scores for each feature
for f in features:
    stats_df = stats_df.withColumn(f"{f}_prev_mean", lag(col(f"{f}_mean"), 1).over(window))
    stats_df = stats_df.withColumn(f"{f}_prev_stddev", lag(col(f"{f}_stddev"), 1).over(window))
    stats_df = stats_df.withColumn(f"{f}_z_score", abs((col(f"{f}_mean") - col(f"{f}_prev_mean")) / col(f"{f}_prev_stddev")))

# Identify the feature with the highest Z-score and its onset date
max_z_scores = []
for f in features:
    max_z_score_row = stats_df.selectExpr(f"max({f}_z_score) as max_z_score", f"event_date as onset_date").orderBy(col("max_z_score").desc()).first()
    if max_z_score_row:
        max_z_scores.append((f, max_z_score_row["max_z_score"], max_z_score_row["onset_date"]))

# Sort by Z-score to find the most drifted feature
max_z_scores.sort(key=lambda x: x[1], reverse=True)
most_drifted_feature, max_z_score, onset_date = max_z_scores[0]

# Write the result to a file in the Volume
result = {"feature": most_drifted_feature, "onset": str(onset_date)}

# Save the result to a local file (for submission)
with open("/tmp/answers.json", "w") as f:
    json.dump(result, f)

# Copy the result to the Volume
spark.createDataFrame([result]).write.mode("overwrite").json("dbfs:/Volumes/workspace/mlpabf81e30/mlpabf81e30_volume/answers.json")

print(f"Most drifted feature: {most_drifted_feature}, Onset date: {onset_date}")