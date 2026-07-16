import hopsworks
import numpy as np
import pandas as pd

proj = hopsworks.login()
fs = proj.get_feature_store()

fg = fs.get_feature_group("prediction_log", version=1)
df = fg.read()
df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
df = df.sort_values("event_time").reset_index(drop=True)

# ---- daily descriptive statistics (platform-side aggregation) ----
df["date"] = df["event_time"].dt.strftime("%Y-%m-%d")
daily = (
    df.groupby("date")["prediction"]
    .agg(n="count", mean_pred="mean", std_pred="std", min_pred="min", max_pred="max")
    .reset_index()
    .sort_values("date")
)
daily["id"] = range(len(daily))
print("DAILY_STATS")
print(daily.to_string())

# ---- single change-point detection on ordered prediction series ----
x = df["prediction"].to_numpy(dtype="float64")
n = len(x)
pre = np.concatenate([[0.0], np.cumsum(x)])
total = pre[-1]
best_k, best_score = None, -1.0
for k in range(1, n):
    m1 = pre[k] / k
    m2 = (total - pre[k]) / (n - k)
    score = (k * (n - k) / n) * (m1 - m2) ** 2
    if score > best_score:
        best_score, best_k = score, k

before_mean = float(pre[best_k] / best_k)
after_mean = float((total - pre[best_k]) / (n - best_k))
onset_ts = df["event_time"].iloc[best_k]
onset_date = onset_ts.strftime("%Y-%m-%d")
print("CHANGEPOINT k=", best_k, "onset_date=", onset_date,
      "before_mean=", before_mean, "after_mean=", after_mean)

# ---- write daily stats FG ----
daily_fg = fs.get_or_create_feature_group(
    name="prediction_daily_stats", version=1,
    description="Daily descriptive statistics of logged predictions",
    primary_key=["date"], online_enabled=False,
)
daily_fg.insert(daily[["id", "date", "n", "mean_pred", "std_pred", "min_pred", "max_pred"]],
                write_options={"wait_for_job": True})

# ---- write shift result FG ----
res = pd.DataFrame([{
    "id": 0,
    "onset_date": onset_date,
    "k_index": int(best_k),
    "before_mean": before_mean,
    "after_mean": after_mean,
    "n_total": int(n),
}])
res_fg = fs.get_or_create_feature_group(
    name="prediction_shift_result", version=1,
    description="Detected distribution-shift onset for logged predictions",
    primary_key=["id"], online_enabled=False,
)
res_fg.insert(res, write_options={"wait_for_job": True})
print("JOB_DONE onset_date=", onset_date)
