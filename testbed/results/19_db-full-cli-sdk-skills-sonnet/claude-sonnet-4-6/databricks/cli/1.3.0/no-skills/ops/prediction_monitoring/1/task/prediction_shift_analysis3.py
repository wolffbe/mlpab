# Databricks notebook source
# COMMAND ----------
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, mean, count
import json
import math

spark = SparkSession.builder.getOrCreate()

# Load data
df = spark.read.option("header", "true").option("inferSchema", "true").csv(
    "/Volumes/workspace/mlpab668870/predictions/prediction_log.csv"
)
df = df.withColumn("ts", col("ts").cast("timestamp"))
df = df.withColumn("prediction", col("prediction").cast("double"))
df = df.withColumn("date", to_date(col("ts")))

daily_stats = df.groupBy("date").agg(
    mean("prediction").alias("daily_mean"),
    count("prediction").alias("cnt")
).orderBy("date")

daily_data = daily_stats.collect()
dates = [r["date"] for r in daily_data]
means = [r["daily_mean"] for r in daily_data]

# Write all daily data to file
lines = ["date,daily_mean"]
for d, m in zip(dates, means):
    lines.append(f"{d},{m:.6f}")

dbutils.fs.put(
    "/Volumes/workspace/mlpab668870/predictions/daily_means.csv",
    "\n".join(lines),
    overwrite=True
)

# Binary segmentation finding best changepoint with improvement metric
best_cost = float('inf')
best_cp = None
total_mean = sum(means) / len(means)
total_cost = sum((x - total_mean)**2 for x in means)

for cp in range(7, len(means) - 7):
    seg1 = means[:cp]
    seg2 = means[cp:]
    m1 = sum(seg1) / len(seg1)
    m2 = sum(seg2) / len(seg2)
    cost = sum((x - m1)**2 for x in seg1) + sum((x - m2)**2 for x in seg2)
    if cost < best_cost:
        best_cost = cost
        best_cp = cp

binseg_date = dates[best_cp]
seg1_mean = sum(means[:best_cp]) / best_cp
seg2_mean = sum(means[best_cp:]) / (len(means) - best_cp)

# Also try with 2-changepoint model
best_cost2 = float('inf')
best_cp1, best_cp2 = None, None
for cp1 in range(7, len(means) - 14):
    for cp2 in range(cp1 + 7, len(means) - 7):
        s1, s2, s3 = means[:cp1], means[cp1:cp2], means[cp2:]
        def seg_cost(s):
            if not s: return 0
            m = sum(s) / len(s)
            return sum((x-m)**2 for x in s)
        cost = seg_cost(s1) + seg_cost(s2) + seg_cost(s3)
        if cost < best_cost2:
            best_cost2 = cost
            best_cp1, best_cp2 = cp1, cp2

# Compute segment means for 2-cp model
s1, s2, s3 = means[:best_cp1], means[best_cp1:best_cp2], means[best_cp2:]
m1_2 = sum(s1)/len(s1)
m2_2 = sum(s2)/len(s2)
m3_2 = sum(s3)/len(s3)

result = {
    "binseg_1cp_onset": str(binseg_date),
    "binseg_1cp_index": best_cp,
    "binseg_2cp_cp1": str(dates[best_cp1]),
    "binseg_2cp_cp2": str(dates[best_cp2]),
    "seg1_mean": seg1_mean,
    "seg2_mean": seg2_mean,
    "m1_2cp": m1_2,
    "m2_2cp": m2_2,
    "m3_2cp": m3_2,
    "n_days": len(means)
}
dbutils.fs.put(
    "/Volumes/workspace/mlpab668870/predictions/result3.json",
    json.dumps(result),
    overwrite=True
)
print("Done")
