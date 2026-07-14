"""Platform-side drift investigation job.

Ingests the daily feature stream into a feature group, computes statistics,
then detects which feature drifted and the onset date via change-point
analysis on per-day statistics. Writes answers.json back to the project.
"""

import json
import os

import hopsworks
import numpy as np
import pandas as pd

proj = hopsworks.login()
fs = proj.get_feature_store()
ds = proj.get_dataset_api()

df = pd.read_csv("/hopsfs/Resources/drift_task/features.csv")
df["event_time"] = pd.to_datetime(df["event_time"])

fg = fs.get_or_create_feature_group(
    name="drift_features",
    version=1,
    primary_key=["entity_id"],
    event_time="event_time",
    statistics_config={"enabled": True, "histograms": True, "correlations": False},
    description="daily feature observations for drift investigation",
)
fg.insert(df, wait=True)
try:
    fg.compute_statistics()
except Exception as e:
    print("compute_statistics failed (non-fatal):", e)

# Read back from the feature group (platform read path) for the analysis.
import time

data = None
for attempt in range(4):
    try:
        data = fg.select_all().read()
        break
    except Exception as e:
        print("fg read attempt", attempt, "failed:", e)
        time.sleep(15)
if data is None or len(data) == 0:
    print("falling back to ingested dataframe for analysis")
    data = df.copy()
data.columns = [c.lower() for c in data.columns]
data["day"] = pd.to_datetime(data["event_time"]).dt.strftime("%Y-%m-%d")

features = ["f1", "f2", "f3", "f4", "f5", "f6"]
daily = data.groupby("day")[features].agg(["mean", "std"]).sort_index()
days = list(daily.index)
n = len(days)
print("days:", n)

best = {"feature": None, "onset": None, "score": -1.0, "kind": None}
for f in features:
    for kind in ("mean", "std"):
        series = daily[(f, kind)].to_numpy()
        for k in range(3, n - 2):
            a, b = series[:k], series[k:]
            pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
            if pooled == 0:
                continue
            stat = abs(b.mean() - a.mean()) / pooled
            if stat > best["score"]:
                best = {"feature": f, "onset": days[k], "score": float(stat), "kind": kind}

print("best:", best)

# Per-day series of the winning feature for the log, to sanity-check onset.
f = best["feature"]
for d, m, s in zip(days, daily[(f, "mean")], daily[(f, "std")]):
    print(f, d, round(m, 4), round(s, 4))

answer = {"feature": best["feature"], "onset": best["onset"]}
with open("answers.json", "w") as fh:
    json.dump(answer, fh)

for target in ("/hopsfs/Resources/submission", "/hopsfs/submission"):
    try:
        os.makedirs(target, exist_ok=True)
        with open(os.path.join(target, "answers.json"), "w") as fh:
            json.dump(answer, fh)
        print("wrote", os.path.join(target, "answers.json"))
    except Exception as e:
        print("write to", target, "failed:", e)

print("ANSWER:", json.dumps(answer))
