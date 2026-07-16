# Databricks notebook source
# COMMAND ----------
import json
import builtins
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, avg, stddev

spark = SparkSession.builder.getOrCreate()

# Load data
df = spark.read.option("header", "true").option("inferSchema", "true").csv(
    "/Volumes/workspace/mlpab098fae/mlpab098fae_data/features.csv"
)

df = df.withColumn("event_date", to_date(col("event_time")))

features = ["f1", "f2", "f3", "f4", "f5", "f6"]

# Compute daily statistics for each feature
daily_stats = df.groupBy("event_date").agg(
    *[avg(f).alias(f"avg_{f}") for f in features],
    *[stddev(f).alias(f"std_{f}") for f in features]
).orderBy("event_date")

stats_pd = daily_stats.toPandas()
stats_pd = stats_pd.sort_values("event_date").reset_index(drop=True)

print("Date range:", stats_pd["event_date"].min(), "to", stats_pd["event_date"].max())
print("Total days:", len(stats_pd))

# COMMAND ----------
# Changepoint detection: for each feature, scan all split points and find the one
# maximizing the t-statistic between before/after mean. The feature with the highest
# score is the drifted one.

n = len(stats_pd)
min_window = 7

results = {}

for feat in features:
    col_name = f"avg_{feat}"
    series = stats_pd[col_name].values

    best_score = 0.0
    best_onset_idx = None

    for t in range(min_window, n - min_window):
        before = series[:t]
        after = series[t:]

        mean_before = sum(before) / len(before)
        mean_after = sum(after) / len(after)

        var_before = sum((x - mean_before)**2 for x in before) / len(before)
        var_after = sum((x - mean_after)**2 for x in after) / len(after)

        pooled_std = (var_before / len(before) + var_after / len(after)) ** 0.5

        if pooled_std < 1e-10:
            continue

        score = abs(mean_after - mean_before) / pooled_std

        if score > best_score:
            best_score = score
            best_onset_idx = t

    if best_onset_idx is not None:
        onset_date = stats_pd["event_date"].iloc[best_onset_idx]
    else:
        onset_date = None

    results[feat] = {"score": best_score, "onset": onset_date}
    print(f"{feat}: score={best_score:.4f}, onset={onset_date}")

# COMMAND ----------
# Find the feature with highest drift score using sorted (avoids PySpark max conflict)
sorted_features = sorted(features, key=lambda f: results[f]["score"], reverse=True)
best_feature = sorted_features[0]
best_onset = results[best_feature]["onset"]

import datetime
if hasattr(best_onset, "strftime"):
    onset_str = best_onset.strftime("%Y-%m-%d")
elif hasattr(best_onset, "isoformat"):
    onset_str = str(best_onset)[:10]
else:
    onset_str = str(best_onset)[:10]

answer = {"feature": best_feature, "onset": onset_str}
print(f"\nFinal answer: {answer}")

# COMMAND ----------
# Save the answer to the volume
answer_json = json.dumps(answer)
dbutils.fs.put("/Volumes/workspace/mlpab098fae/mlpab098fae_data/answers.json", answer_json, overwrite=True)
print(f"Saved answer: {answer_json}")
