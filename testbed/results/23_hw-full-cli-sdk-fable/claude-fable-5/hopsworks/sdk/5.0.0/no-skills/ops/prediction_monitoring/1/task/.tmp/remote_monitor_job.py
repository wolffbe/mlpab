# Runs ON the Hopsworks cluster as a Python job.
# Ingests the prediction log into a feature group, computes per-day statistics,
# detects the distribution shift onset, and writes results to the Datasets.
import json
import math

import hopsworks
import pandas as pd

proj = hopsworks.login()
fs = proj.get_feature_store()
dataset_api = proj.get_dataset_api()

df = pd.read_csv("/hopsfs/Resources/prediction_log.csv")
df["ts"] = pd.to_datetime(df["ts"], utc=True)
df["pred_id"] = list(range(1, len(df) + 1))

fg = fs.get_or_create_feature_group(
    name="prediction_log",
    version=1,
    description="Deployed model logged predictions for monitoring",
    primary_key=["pred_id"],
    event_time="ts",
    online_enabled=False,
)
fg.insert(df, wait=True)

# statistics on the feature group (platform monitoring capability)
try:
    fg.compute_statistics()
except Exception as e:
    print("compute_statistics failed:", e)

# read back from the feature store to base monitoring on platform-stored data
import time

stored = None
for attempt in range(6):
    try:
        stored = fg.read()
        break
    except Exception as e:
        print(f"fg.read attempt {attempt} failed: {e}")
        time.sleep(15)
if stored is None:
    print("falling back to inserted dataframe")
    stored = df[["ts", "prediction"]].copy()
stored["ts"] = pd.to_datetime(stored["ts"], utc=True)
stored["day"] = stored["ts"].dt.strftime("%Y-%m-%d")
daily = stored.groupby("day")["prediction"].agg(["mean", "std", "count"]).reset_index()
daily = daily.sort_values("day").reset_index(drop=True)

# change-point detection on daily means: pick split maximizing two-sample t-statistic
means = daily["mean"].tolist()
n = len(means)
best_k, best_t = None, -1.0
for k in range(3, n - 2):  # 'after' segment starts at index k
    a = means[:k]
    b = means[k:]
    ma = sum(a) / len(a)
    mb = sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a) / max(len(a) - 1, 1)
    vb = sum((x - mb) ** 2 for x in b) / max(len(b) - 1, 1)
    se = math.sqrt(va / len(a) + vb / len(b))
    if se == 0:
        continue
    t = abs(ma - mb) / se
    if t > best_t:
        best_t, best_k = t, k

onset = daily.loc[best_k, "day"]
print("ONSET:", onset, "t=", best_t)
print("DAILY_MEANS:", json.dumps(daily[["day", "mean"]].round(4).to_dict(orient="records")))

result = {
    "onset": onset,
    "t_stat": best_t,
    "daily": daily.to_dict(orient="records"),
}
with open("/tmp/monitor_results.json", "w") as f:
    json.dump(result, f)
dataset_api.upload("/tmp/monitor_results.json", "Resources", overwrite=True)

with open("/tmp/answers.json", "w") as f:
    json.dump({"onset": onset}, f)
dataset_api.upload("/tmp/answers.json", "Resources", overwrite=True)
print("uploaded results to Resources")
