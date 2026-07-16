# Databricks notebook source
# MAGIC %md
# MAGIC ## Feature Drift Detection
# MAGIC 
# MAGIC This notebook:
# MAGIC 1. Creates a table from the feature data.
# MAGIC 2. Computes daily statistics for each feature.
# MAGIC 3. Identifies the feature with the most significant distribution shift.

# COMMAND ----------

# Create the table from the CSV file in the Volume
spark.sql("""
CREATE TABLE IF NOT EXISTS workspace.mlpabf81e30.mlpabf81e30_features
USING CSV
OPTIONS (
    header = "true",
    path = "dbfs:/Volumes/workspace/mlpabf81e30/mlpabf81e30_volume/features.csv"
)
""")

# COMMAND ----------

# Compute daily statistics for each feature
from pyspark.sql.functions import col, to_date, mean, stddev, count

# Read the data
df = spark.table("workspace.mlpabf81e30.mlpabf81e30_features")

# Extract date from event_time
df = df.withColumn("event_date", to_date(col("event_time")))

# Compute daily statistics for each feature
features = ["f1", "f2", "f3", "f4", "f5", "f6"]
stats_df = df.groupBy("event_date").agg(
    *[mean(col(f)).alias(f"{f}_mean") for f in features],
    *[stddev(col(f)).alias(f"{f}_stddev") for f in features],
    count("*").alias("count")
).orderBy("event_date")

# Show the statistics
stats_df.display()

# COMMAND ----------

# Identify the feature with the most significant drift
# We will compute the rolling Z-score for each feature's mean
from pyspark.sql.window import Window
from pyspark.sql.functions import lag, abs

# Define a window for rolling statistics
window = Window.orderBy("event_date")

# Compute rolling mean and stddev for each feature
for f in features:
    stats_df = stats_df.withColumn(f"{f}_prev_mean", lag(col(f"{f}_mean"), 1).over(window))
    stats_df = stats_df.withColumn(f"{f}_prev_stddev", lag(col(f"{f}_stddev"), 1).over(window))
    stats_df = stats_df.withColumn(f"{f}_z_score", abs((col(f"{f}_mean") - col(f"{f}_prev_mean")) / col(f"{f}_prev_stddev")))

# Show the Z-scores
stats_df.select([col(f"{f}_z_score") for f in features] + ["event_date"]).display()

# COMMAND ----------

# Identify the feature with the highest Z-score and its onset date
max_z_scores = []
for f in features:
    max_z_score = stats_df.selectExpr(f"max({f}_z_score) as max_z_score", f"event_date as onset_date").orderBy(col("max_z_score").desc()).first()
    if max_z_score:
        max_z_scores.append((f, max_z_score["max_z_score"], max_z_score["onset_date"]))

# Sort by Z-score to find the most drifted feature
max_z_scores.sort(key=lambda x: x[1], reverse=True)
most_drifted_feature, max_z_score, onset_date = max_z_scores[0]

# Write the result to a file in the Volume
result = {"feature": most_drifted_feature, "onset": str(onset_date)}

# Save the result to a file in the Volume
with open("/dbfs/Volumes/workspace/mlpabf81e30/mlpabf81e30_volume/result.json", "w") as f:
    import json
    json.dump(result, f)

# Display the result
print(f"Most drifted feature: {most_drifted_feature}, Onset date: {onset_date}")