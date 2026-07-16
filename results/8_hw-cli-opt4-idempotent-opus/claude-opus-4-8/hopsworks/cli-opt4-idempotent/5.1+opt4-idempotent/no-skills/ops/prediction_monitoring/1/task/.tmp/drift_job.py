"""Prediction-distribution drift monitoring job.

Runs on the Hopsworks cluster. Reads the logged predictions feature group,
computes per-day distribution statistics (the model-monitoring step), detects
the single change point where the prediction distribution shifted, and
persists the daily statistics + detected onset back to the feature store.
"""
import hopsworks
from pyspark.sql import functions as F
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()

fg = fs.get_feature_group("prediction_log", version=1)
df = fg.read()

daily = (
    df.withColumn("day", F.substring(F.col("ts"), 1, 10))
      .groupBy("day")
      .agg(
          F.count(F.lit(1)).alias("n"),
          F.avg("prediction").alias("mean"),
          F.stddev("prediction").alias("sd"),
          F.sum("prediction").alias("s"),
      )
      .orderBy("day")
)
rows = daily.collect()

days = [r["day"] for r in rows]
n = [int(r["n"]) for r in rows]
s = [float(r["s"]) for r in rows]
mean = [float(r["mean"]) for r in rows]
sd = [float(r["sd"]) if r["sd"] is not None else 0.0 for r in rows]

N = len(days)
pn = [0] * (N + 1)
ps = [0.0] * (N + 1)
for i in range(N):
    pn[i + 1] = pn[i] + n[i]
    ps[i + 1] = ps[i] + s[i]
totN = pn[N]
totS = ps[N]

# Single change point: split that maximizes between-segment sum of squares.
best_k = None
best_score = -1.0
for k in range(1, N):
    nA = pn[k]
    sA = ps[k]
    nB = totN - nA
    sB = totS - sA
    if nA == 0 or nB == 0:
        continue
    muA = sA / nA
    muB = sB / nB
    score = (nA * nB / float(totN)) * (muA - muB) ** 2
    if score > best_score:
        best_score = score
        best_k = k

onset = days[best_k]
mu_before = ps[best_k] / pn[best_k]
mu_after = (totS - ps[best_k]) / (totN - pn[best_k])

print("DAILY_MEANS_START")
for d, m, c in zip(days, mean, n):
    print(f"  {d} n={c} mean={m:.4f}")
print("DAILY_MEANS_END")
print(f"MEAN_BEFORE={mu_before:.4f} MEAN_AFTER={mu_after:.4f}")
print(f"DETECTED_ONSET={onset}")

# Persist daily statistics (the monitoring artifact).
stats_pdf = pd.DataFrame({"day": days, "n": n, "mean": mean, "sd": sd})
sfg = fs.get_or_create_feature_group(
    "prediction_daily_stats",
    version=1,
    primary_key=["day"],
    description="Daily prediction distribution statistics for monitoring",
    online_enabled=False,
)
sfg.insert(stats_pdf)

# Persist the detected shift onset.
onset_pdf = pd.DataFrame({
    "metric": ["onset"],
    "onset": [onset],
    "mean_before": [float(mu_before)],
    "mean_after": [float(mu_after)],
})
ofg = fs.get_or_create_feature_group(
    "prediction_drift_onset",
    version=1,
    primary_key=["metric"],
    description="Detected prediction distribution shift onset",
    online_enabled=False,
)
ofg.insert(onset_pdf)

print("DONE")
