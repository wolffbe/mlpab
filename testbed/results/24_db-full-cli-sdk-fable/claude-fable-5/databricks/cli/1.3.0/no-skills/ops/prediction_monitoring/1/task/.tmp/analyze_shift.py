# Databricks notebook source
from pyspark.sql import functions as F

df = (spark.read.option("header", True)
      .csv("/Volumes/workspace/mlpabf9b9a1/predlogs/prediction_log.csv")
      .select(F.to_timestamp("ts").alias("ts"), F.col("prediction").cast("double").alias("prediction")))

df.write.mode("overwrite").saveAsTable("workspace.mlpabf9b9a1.prediction_log")

daily = (df.groupBy(F.to_date("ts").alias("day"))
           .agg(F.mean("prediction").alias("mean_pred"),
                F.stddev("prediction").alias("std_pred"),
                F.count("*").alias("n"))
           .orderBy("day"))
daily.write.mode("overwrite").saveAsTable("workspace.mlpabf9b9a1.prediction_daily_stats")

rows = daily.collect()
days = [r["day"] for r in rows]
means = [r["mean_pred"] for r in rows]
stds = [r["std_pred"] for r in rows]

# Changepoint: split the daily-mean series at each candidate day k (segment 2
# starts at k) and minimize total within-segment sum of squared errors.
def sse(xs):
    if not xs:
        return 0.0
    m = sum(xs) / len(xs)
    return sum((x - m) ** 2 for x in xs)

best_k, best_cost = None, float("inf")
for k in range(1, len(means)):
    cost = sse(means[:k]) + sse(means[k:])
    if cost < best_cost:
        best_cost, best_k = cost, k

onset_mean = str(days[best_k])

# Same on daily stddev in case the shift is in variance rather than mean
best_k2, best_cost2 = None, float("inf")
for k in range(1, len(stds)):
    cost = sse(stds[:k]) + sse(stds[k:])
    if cost < best_cost2:
        best_cost2, best_k2 = cost, k
onset_std = str(days[best_k2])

pre = means[:best_k]; post = means[best_k:]
mean_jump = abs(sum(post)/len(post) - sum(pre)/len(pre))
pre_s = stds[:best_k2]; post_s = stds[best_k2:]
std_jump = abs(sum(post_s)/len(post_s) - sum(pre_s)/len(pre_s))

result = [(onset_mean, float(mean_jump), onset_std, float(std_jump))]
spark.createDataFrame(result, "onset_mean string, mean_jump double, onset_std string, std_jump double") \
     .write.mode("overwrite").saveAsTable("workspace.mlpabf9b9a1.shift_result")

print("ONSET_MEAN", onset_mean, "jump", mean_jump)
print("ONSET_STD", onset_std, "jump", std_jump)
for d, m, s in zip(days, means, stds):
    print(d, round(m, 4), round(s, 4))
