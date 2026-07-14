# Databricks notebook source
import json
import os
from pyspark.sql import functions as F

schema = "workspace.mlpab2964d0"
catalog, sch = schema.split(".")
csv_path = f"/Volumes/{catalog}/{sch}/prediction_logs/prediction_log.csv"

df = (spark.read.option("header", True).csv(csv_path)
      .select(F.to_timestamp("ts").alias("ts"),
              F.col("prediction").cast("double").alias("prediction")))
df.write.mode("overwrite").saveAsTable(f"{schema}.prediction_log")

# Daily monitoring statistics
daily = (df.groupBy(F.to_date("ts").alias("day"))
           .agg(F.count("*").alias("n"),
                F.avg("prediction").alias("mean_pred"),
                F.stddev("prediction").alias("std_pred"),
                F.min("prediction").alias("min_pred"),
                F.max("prediction").alias("max_pred"),
                F.expr("percentile(prediction, 0.5)").alias("median_pred"))
           .orderBy("day"))
daily.write.mode("overwrite").saveAsTable(f"{schema}.prediction_daily_stats")

rows = daily.collect()

# Change-point detection: for each candidate onset day d, score the absolute
# difference between the mean of all predictions on/after d and the mean
# before d (computed from daily sums/counts).
total_sum = sum(r["mean_pred"] * r["n"] for r in rows)
total_n = sum(r["n"] for r in rows)
best_day, best_score = None, -1.0
cum_sum, cum_n = 0.0, 0
for r in rows[:-1]:
    cum_sum += r["mean_pred"] * r["n"]
    cum_n += r["n"]
    after_n = total_n - cum_n
    score = abs((total_sum - cum_sum) / after_n - cum_sum / cum_n)
    # weight by sqrt of balanced sizes (standard CUSUM-like normalization)
    weight = (cum_n * after_n / total_n) ** 0.5
    wscore = score * weight
    if wscore > best_score:
        best_score = wscore
        best_day = r["day"]  # last day of the "before" segment

onset = None
for i, r in enumerate(rows):
    if r["day"] == best_day:
        onset = rows[i + 1]["day"]
        break

daily_summary = [
    {"day": str(r["day"]), "n": r["n"],
     "mean": round(r["mean_pred"], 4), "std": round(r["std_pred"], 4)}
    for r in rows
]
result = {"onset": str(onset), "score": best_score, "daily": daily_summary}
spark.createDataFrame([(json.dumps(result),)], "result string") \
    .write.mode("overwrite").saveAsTable(f"{schema}.shift_result")
dbutils.notebook.exit(json.dumps(result))
