# Databricks notebook source
# COMMAND ----------
from pyspark.sql import functions as F

df = spark.read.csv(
    "/Volumes/workspace/mlpabe8ecdb/mlpabe8ecdb_drift_data/features.csv",
    header=True,
    inferSchema=True
)
df = df.withColumn("event_date", F.to_date(F.col("event_time")))

features = ["f1", "f2", "f3", "f4", "f5", "f6"]
agg_exprs = []
for feat in features:
    agg_exprs.append(F.mean(F.col(feat)).alias("mean_" + feat))
    agg_exprs.append(F.stddev(F.col(feat)).alias("std_" + feat))

daily_stats = df.groupBy("event_date").agg(*agg_exprs).orderBy("event_date")
pdf = daily_stats.toPandas()
pdf = pdf.sort_values("event_date").reset_index(drop=True)

import numpy as np

n_days = len(pdf)
results = {}

for feat in features:
    means = pdf["mean_" + feat].values
    best_split = None
    best_score = 0.0
    for split in range(10, n_days - 10):
        pre_mean = float(np.mean(means[:split]))
        post_mean = float(np.mean(means[split:]))
        pre_std = float(np.std(means[:split]))
        post_std = float(np.std(means[split:]))
        pooled_std = float(np.sqrt((pre_std**2 + post_std**2) / 2.0)) + 1e-10
        score = abs(post_mean - pre_mean) / pooled_std
        if score > best_score:
            best_score = score
            best_split = split
    onset_date = str(pdf["event_date"].iloc[best_split]) if best_split is not None else "unknown"
    results[feat] = {
        "score": float(best_score),
        "onset_date": onset_date,
        "split_idx": int(best_split) if best_split is not None else 0,
        "pre_mean": float(np.mean(means[:best_split])) if best_split else 0.0,
        "post_mean": float(np.mean(means[best_split:])) if best_split else 0.0,
    }
    print(feat + ": score=" + str(round(best_score, 3)) + " onset=" + onset_date)

best_feature = max(results, key=lambda f: results[f]["score"])
answer_onset = results[best_feature]["onset_date"][:10]
print("ANSWER FEATURE: " + best_feature)
print("ANSWER ONSET: " + answer_onset)

import json
answer = {"feature": best_feature, "onset": answer_onset}
print("ANSWER JSON: " + json.dumps(answer))

all_rows = []
for feat, r in results.items():
    all_rows.append((feat, float(r["score"]), str(r["onset_date"]), int(r["split_idx"]),
                     float(r["pre_mean"]), float(r["post_mean"])))

from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType
schema = StructType([
    StructField("feature", StringType()),
    StructField("score", FloatType()),
    StructField("onset_date", StringType()),
    StructField("split_idx", IntegerType()),
    StructField("pre_mean", FloatType()),
    StructField("post_mean", FloatType()),
])
results_df = spark.createDataFrame(all_rows, schema)
results_df.write.format("delta").mode("overwrite").saveAsTable(
    "workspace.mlpabe8ecdb.mlpabe8ecdb_drift_results"
)

answer_rows = [(best_feature, answer_onset)]
answer_schema = StructType([
    StructField("feature", StringType()),
    StructField("onset", StringType()),
])
answer_df = spark.createDataFrame(answer_rows, answer_schema)
answer_df.write.format("delta").mode("overwrite").saveAsTable(
    "workspace.mlpabe8ecdb.mlpabe8ecdb_drift_answer"
)
print("Saved results to Delta tables.")
