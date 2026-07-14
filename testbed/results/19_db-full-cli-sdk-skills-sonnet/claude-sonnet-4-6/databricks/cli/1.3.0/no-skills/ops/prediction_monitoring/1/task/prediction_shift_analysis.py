# Databricks notebook source
# COMMAND ----------
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, avg, stddev, count, mean, lit
from pyspark.sql.types import DoubleType
import json

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------
# Load the prediction log
df = spark.read.option("header", "true").option("inferSchema", "true").csv(
    "/Volumes/workspace/mlpab668870/predictions/prediction_log.csv"
)
df = df.withColumn("ts", col("ts").cast("timestamp"))
df = df.withColumn("prediction", col("prediction").cast("double"))
df = df.withColumn("date", to_date(col("ts")))

# COMMAND ----------
# Compute daily statistics
daily_stats = df.groupBy("date").agg(
    mean("prediction").alias("daily_mean"),
    stddev("prediction").alias("daily_std"),
    count("prediction").alias("cnt")
).orderBy("date")

daily_stats.show(100)

# COMMAND ----------
# Detect distribution shift using sliding window approach
# Compare each day's mean against the baseline (first 14 days)
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, abs as spark_abs

# Collect to driver for analysis
daily_data = daily_stats.collect()

# Compute baseline stats (first 14 days)
baseline_days = 14
baseline = daily_data[:baseline_days]
baseline_means = [r["daily_mean"] for r in baseline]
baseline_mean = sum(baseline_means) / len(baseline_means)
baseline_std = (sum((x - baseline_mean)**2 for x in baseline_means) / len(baseline_means)) ** 0.5

print(f"Baseline mean: {baseline_mean:.4f}")
print(f"Baseline std: {baseline_std:.4f}")

# COMMAND ----------
# Find shift onset using cumulative mean comparison
# Look for sustained deviation from baseline
window_size = 3
threshold = 2.0  # standard deviations from baseline

print("\nSearching for distribution shift...")
print(f"Threshold: baseline_mean +/- {threshold} * baseline_std = {baseline_mean - threshold*baseline_std:.4f} to {baseline_mean + threshold*baseline_std:.4f}")

shift_onset = None
consecutive_shifts = 0
required_consecutive = 3

for i in range(baseline_days, len(daily_data)):
    day_mean = daily_data[i]["daily_mean"]
    deviation = abs(day_mean - baseline_mean) / (baseline_std + 1e-10)

    if deviation > threshold:
        consecutive_shifts += 1
        if consecutive_shifts >= required_consecutive and shift_onset is None:
            # Find the start of this consecutive run
            shift_onset = daily_data[i - required_consecutive + 1]["date"]
            print(f"Shift detected! Onset: {shift_onset}")
    else:
        consecutive_shifts = 0

print(f"\nShift onset: {shift_onset}")

# COMMAND ----------
# Alternative: use cumulative mean and look for breakpoint
# Compute rolling mean and find where it departs from baseline
print("\nDetailed daily analysis:")
for row in daily_data:
    deviation = (row["daily_mean"] - baseline_mean) / (baseline_std + 1e-10)
    print(f"  {row['date']}: mean={row['daily_mean']:.4f}, dev={deviation:.2f}sigma")

# COMMAND ----------
# Write result to volume
onset_str = str(shift_onset)

result = {"onset": onset_str}
result_json = json.dumps(result)

# Write JSON to volume
dbutils.fs.put("/Volumes/workspace/mlpab668870/predictions/result.json", result_json, overwrite=True)

print(f"\nResult written: {result_json}")

# COMMAND ----------
# Also store as a Delta table for easy readback
spark.createDataFrame([(onset_str,)], ["onset"]).write.mode("overwrite").saveAsTable("workspace.mlpab668870.shift_result")
print("Result saved to table workspace.mlpab668870.shift_result")
