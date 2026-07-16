# Databricks notebook source
import json

df = (spark.read.format("csv").option("header", True).option("inferSchema", True)
      .load("/Volumes/workspace/mlpab002289/drift_data/features.csv"))

from pyspark.sql import functions as F

feats = ["f1", "f2", "f3", "f4", "f5", "f6"]
daily = (df.withColumn("day", F.to_date("event_time"))
           .groupBy("day")
           .agg(*[F.avg(f).alias(f + "_mean") for f in feats],
                *[F.stddev(f).alias(f + "_std") for f in feats])
           .orderBy("day"))

rows = daily.collect()
days = [str(r["day"]) for r in rows]
n_base = 14  # baseline window: first 14 days

result = {}
for f in feats:
    means = [r[f + "_mean"] for r in rows]
    stds = [r[f + "_std"] for r in rows]
    for stat, series in (("mean", means), ("std", stds)):
        base = series[:n_base]
        mu = sum(base) / len(base)
        var = sum((x - mu) ** 2 for x in base) / (len(base) - 1)
        sd = var ** 0.5 if var > 0 else 1e-9
        z = [(x - mu) / sd for x in series]
        # onset: first day where |z|>4 and the following 5 days all have |z|>3
        onset = None
        for i in range(n_base, len(z)):
            if abs(z[i]) > 4 and all(abs(zz) > 3 for zz in z[i + 1:i + 6]):
                onset = i
                break
        # severity: mean |z| after onset
        if onset is not None:
            sev = sum(abs(zz) for zz in z[onset:]) / len(z[onset:])
            result[f + "_" + stat] = {"onset": days[onset], "severity": sev}

# report all per-day z-summary for transparency
report = {
    "candidates": result,
    "daily_days": days,
}

# pick the strongest candidate
best = max(result.items(), key=lambda kv: kv[1]["severity"]) if result else None
answer = {"feature": best[0].split("_")[0], "onset": best[0] and best[1]["onset"]} if best else {}
report["answer"] = answer

with open("/Volumes/workspace/mlpab002289/drift_data/answers.json", "w") as fh:
    json.dump(answer, fh)
with open("/Volumes/workspace/mlpab002289/drift_data/drift_report.json", "w") as fh:
    json.dump(report, fh, indent=2)

print(json.dumps(report, indent=2)[:4000])
