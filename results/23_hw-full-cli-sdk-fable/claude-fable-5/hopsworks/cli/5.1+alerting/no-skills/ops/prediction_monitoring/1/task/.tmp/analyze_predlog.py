"""Analyze pred_log feature group for a distribution shift — runs as a
Hopsworks PYTHON job (cluster-side). Computes daily statistics and finds the
change point by minimizing two-segment squared error over daily means."""
import json
import hopsworks
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()

fg = fs.get_feature_group("pred_log", version=1)
df = fg.read()
df["ts"] = pd.to_datetime(df["ts"])
df["day"] = df["ts"].dt.date

daily = df.groupby("day")["prediction"].agg(["count", "mean", "std", "median"]).reset_index()
daily = daily.sort_values("day").reset_index(drop=True)
print("=== DAILY STATS ===")
for _, r in daily.iterrows():
    print(f"{r['day']} n={int(r['count'])} mean={r['mean']:.4f} std={r['std']:.4f} median={r['median']:.4f}")

# Two-segment change-point detection on daily means
means = daily["mean"].values
best_k, best_sse = None, None
for k in range(1, len(means)):
    a, b = means[:k], means[k:]
    sse = ((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum()
    if best_sse is None or sse < best_sse:
        best_sse, best_k = sse, k

onset_day = daily["day"].iloc[best_k]
pre = means[:best_k]
post = means[best_k:]
print("=== CHANGE POINT ===")
print(f"onset={onset_day} pre_mean={pre.mean():.4f} post_mean={post.mean():.4f}")

answer = {"onset": str(onset_day)}
print("ANSWER_JSON:" + json.dumps(answer))

# Persist the answer on the platform
with open("answers.json", "w") as f:
    json.dump(answer, f)
dataset_api = project.get_dataset_api()
dataset_api.upload("answers.json", "Resources", overwrite=True)
print("Uploaded answers.json to Resources")
