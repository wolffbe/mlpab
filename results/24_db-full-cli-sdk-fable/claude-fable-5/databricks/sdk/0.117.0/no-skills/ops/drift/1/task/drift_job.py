import json
import sys

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.getOrCreate()
vol = sys.argv[1]  # /Volumes/<cat>/<sch>/drift_vol

df = spark.read.csv(f"{vol}/features.csv", header=True, inferSchema=True)
feats = ["f1", "f2", "f3", "f4", "f5", "f6"]
daily = (
    df.withColumn("d", F.to_date("event_time"))
    .groupBy("d")
    .agg(*[F.avg(f).alias(f"{f}_mean") for f in feats],
         *[F.stddev(f).alias(f"{f}_std") for f in feats])
    .orderBy("d")
    .collect()
)

dates = [str(r["d"]) for r in daily]
best = None
for f in feats:
    means = [r[f"{f}_mean"] for r in daily]
    base = means[:14]
    bmean = sum(base) / len(base)
    bstd = (sum((x - bmean) ** 2 for x in base) / (len(base) - 1)) ** 0.5
    # first date where the daily mean deviates > 5 sigma (of baseline daily-mean
    # variation) and stays deviated for at least 5 consecutive days
    onset, score = None, 0.0
    for i in range(14, len(means)):
        z = abs(means[i] - bmean) / (bstd + 1e-12)
        if z > 5:
            window = [abs(m - bmean) / (bstd + 1e-12) for m in means[i:i + 5]]
            if all(w > 3 for w in window):
                onset = dates[i]
                score = sum(abs(m - bmean) / (bstd + 1e-12) for m in means[i:]) / (len(means) - i)
                break
    if onset and (best is None or score > best[2]):
        best = (f, onset, score)

result = {"feature": best[0], "onset": best[1]} if best else {"feature": None, "onset": None}
diag = {f: [round(r[f"{f}_mean"], 3) for r in daily] for f in feats}
with open(f"{vol}/answers.json", "w") as out:
    json.dump(result, out)
with open(f"{vol}/daily_means.json", "w") as out:
    json.dump({"dates": dates, "means": diag, "score": best[2] if best else None}, out)
print(json.dumps(result))
