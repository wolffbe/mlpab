# Databricks notebook source
# COMMAND ----------
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, avg, stddev, count, mean, lit
from pyspark.sql.types import DoubleType
import json
import math

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------
# Load data
df = spark.read.option("header", "true").option("inferSchema", "true").csv(
    "/Volumes/workspace/mlpab668870/predictions/prediction_log.csv"
)
df = df.withColumn("ts", col("ts").cast("timestamp"))
df = df.withColumn("prediction", col("prediction").cast("double"))
df = df.withColumn("date", to_date(col("ts")))

# Compute daily mean
daily_stats = df.groupBy("date").agg(
    mean("prediction").alias("daily_mean"),
    stddev("prediction").alias("daily_std"),
    count("prediction").alias("cnt")
).orderBy("date")

daily_data = daily_stats.collect()
dates = [r["date"] for r in daily_data]
means = [r["daily_mean"] for r in daily_data]
counts = [r["cnt"] for r in daily_data]

print(f"Total days: {len(daily_data)}")
print(f"Date range: {dates[0]} to {dates[-1]}")
print(f"Mean range: {min(means):.4f} to {max(means):.4f}")

# COMMAND ----------
# Method 1: CUSUM (Cumulative Sum Control Chart)
# Detects when the running mean deviates significantly from target
baseline_n = 14
baseline_mean = sum(means[:baseline_n]) / baseline_n
baseline_vars = [(x - baseline_mean)**2 for x in means[:baseline_n]]
baseline_std = math.sqrt(sum(baseline_vars) / baseline_n)

print(f"\nBaseline (first {baseline_n} days):")
print(f"  Mean: {baseline_mean:.4f}")
print(f"  Std: {baseline_std:.4f}")

# CUSUM parameters
k = 0.5  # allowance (sensitivity)
h = 5.0  # threshold

cusum_pos = 0.0
cusum_neg = 0.0
cusum_onset = None

for i in range(baseline_n, len(daily_data)):
    z = (means[i] - baseline_mean) / (baseline_std + 1e-10)
    cusum_pos = max(0, cusum_pos + z - k)
    cusum_neg = min(0, cusum_neg + z + k)

    if (cusum_pos > h or cusum_neg < -h) and cusum_onset is None:
        cusum_onset = dates[i]
        print(f"CUSUM shift detected at {dates[i]}: cusum_pos={cusum_pos:.2f}, cusum_neg={cusum_neg:.2f}")

print(f"CUSUM onset: {cusum_onset}")

# COMMAND ----------
# Method 2: Sliding window t-test
# Compare window of N days against baseline
window_size = 7
t_threshold = 3.0

def welch_t_stat(m1, s1, n1, m2, s2, n2):
    """Welch's t-statistic"""
    if s1 == 0 and s2 == 0:
        return 0.0
    se = math.sqrt((s1**2 / n1) + (s2**2 / n2))
    if se == 0:
        return 0.0
    return abs(m1 - m2) / se

baseline_m = sum(means[:baseline_n]) / baseline_n
baseline_s = math.sqrt(sum((x - baseline_m)**2 for x in means[:baseline_n]) / baseline_n) + 1e-10

ttest_onset = None
for i in range(baseline_n, len(daily_data) - window_size + 1):
    window = means[i:i+window_size]
    w_mean = sum(window) / len(window)
    w_std = math.sqrt(sum((x - w_mean)**2 for x in window) / len(window)) + 1e-10

    t_stat = welch_t_stat(baseline_m, baseline_s, baseline_n, w_mean, w_std, window_size)

    if t_stat > t_threshold and ttest_onset is None:
        ttest_onset = dates[i]
        print(f"T-test shift detected starting at {dates[i]}: t={t_stat:.2f}, window_mean={w_mean:.4f}")

print(f"T-test onset: {ttest_onset}")

# COMMAND ----------
# Method 3: Binary segmentation - find the single best changepoint
# Minimize within-segment variance
best_cost = float('inf')
best_cp = None

total_mean = sum(means) / len(means)
total_cost = sum((x - total_mean)**2 for x in means)

print(f"\nBinary segmentation (find single best changepoint):")
for cp in range(10, len(means) - 10):
    seg1 = means[:cp]
    seg2 = means[cp:]

    m1 = sum(seg1) / len(seg1)
    m2 = sum(seg2) / len(seg2)

    cost = sum((x - m1)**2 for x in seg1) + sum((x - m2)**2 for x in seg2)

    if cost < best_cost:
        best_cost = cost
        best_cp = cp

best_date = dates[best_cp]
seg1_mean = sum(means[:best_cp]) / best_cp
seg2_mean = sum(means[best_cp:]) / (len(means) - best_cp)
print(f"Best changepoint: {best_date} (index {best_cp})")
print(f"  Before: mean={seg1_mean:.4f}")
print(f"  After:  mean={seg2_mean:.4f}")
print(f"  Improvement: {total_cost - best_cost:.4f}")

# COMMAND ----------
# Print all three results
print("\n=== SUMMARY ===")
print(f"CUSUM onset:             {cusum_onset}")
print(f"T-test onset:            {ttest_onset}")
print(f"Binary segmentation:     {best_date}")

# Use binary segmentation as the most reliable
onset = str(best_date)
print(f"\nFinal answer: {onset}")

# COMMAND ----------
# Write all results to volume for comparison
results = {
    "onset": onset,
    "cusum_onset": str(cusum_onset),
    "ttest_onset": str(ttest_onset),
    "binseg_onset": str(best_date),
    "baseline_mean": baseline_mean,
    "seg1_mean": seg1_mean,
    "seg2_mean": seg2_mean
}
result_json = json.dumps(results)
dbutils.fs.put("/Volumes/workspace/mlpab668870/predictions/result2.json", result_json, overwrite=True)
print(f"Results written: {result_json}")

# COMMAND ----------
# Print detailed daily data around the detected changepoint
print("\nDaily means around detected changepoint:")
for i in range(max(0, best_cp - 7), min(len(daily_data), best_cp + 7)):
    marker = " <<< CHANGEPOINT" if i == best_cp else ""
    print(f"  {dates[i]}: {means[i]:.4f}{marker}")
