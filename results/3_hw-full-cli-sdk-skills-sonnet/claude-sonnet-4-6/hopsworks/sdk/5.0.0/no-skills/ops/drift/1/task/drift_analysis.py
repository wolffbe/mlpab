import hopsworks
import pandas as pd
import json

print("Connecting to Hopsworks...")
project = hopsworks.login()
fs = project.get_feature_store()

fg = fs.get_or_create_feature_group(
    name="drift_weekly_features",
    version=1,
    primary_key=["entity_id"],
    event_time="event_time",
)

print("Reading data from feature group...")
df = fg.read()
print(f"Shape: {df.shape}")

# Compute daily means for all features
df["date"] = pd.to_datetime(df["event_time"]).dt.date
feature_names = ["f1", "f2", "f3", "f4", "f5", "f6"]
daily_means = df.groupby("date")[feature_names].mean()

print("\n=== Daily means for f3 (the drifted feature) ===")
print(daily_means["f3"].to_string())

print("\n=== Around the drift point (Feb-Mar 2026) ===")
onset_area = daily_means.loc[
    (daily_means.index >= pd.Timestamp("2026-02-20").date()) &
    (daily_means.index <= pd.Timestamp("2026-03-10").date())
]
print(onset_area[["f3"]].to_string())

# Find day-over-day changes for f3
f3_daily = daily_means["f3"]
f3_diff = f3_daily.diff()
print("\n=== Day-over-day changes for f3 ===")
print(f3_diff.to_string())

# Find the onset: first day where f3 changes significantly
baseline_mean = f3_daily.iloc[:30].mean()  # first 30 days baseline
baseline_std = f3_daily.iloc[:30].std()
print(f"\nBaseline mean (first 30 days): {baseline_mean:.4f}")
print(f"Baseline std: {baseline_std:.4f}")
print(f"Threshold (2 std): {baseline_mean + 2*baseline_std:.4f}")

# Find first day that exceeds threshold
threshold = baseline_mean + 2 * baseline_std
above_threshold = f3_daily[f3_daily > threshold]
print(f"\nFirst day exceeding threshold: {above_threshold.index[0]}")
print(f"Value on that day: {above_threshold.iloc[0]:.4f}")

# Find the exact onset by looking at when the big jump happens
print("\n=== All days around drift ===")
for i, (date, val) in enumerate(f3_daily.items()):
    if i > 0:
        prev_val = f3_daily.iloc[i-1]
        change = val - prev_val
        if abs(change) > 1.0:
            print(f"BIG CHANGE: {date}: {val:.4f} (prev: {prev_val:.4f}, change: {change:.4f})")
