import hopsworks
import pandas as pd
import os
import json
import math

# Connect to Hopsworks
print("Connecting to Hopsworks...")
project = hopsworks.login()
fs = project.get_feature_store()
print("Connected to project:", project.name)

# Read data
print("Reading data...")
df = pd.read_csv("data/features.csv")
df["event_time"] = pd.to_datetime(df["event_time"])
print(f"Data shape: {df.shape}")
print(f"Date range: {df['event_time'].min()} to {df['event_time'].max()}")

features = ["f1", "f2", "f3", "f4", "f5", "f6"]
dates = sorted(df["event_time"].dt.date.unique())
print(f"Number of dates: {len(dates)}")

# --- Create or get the feature group ---
FG_NAME = "drift_investigation_fg"
FG_VERSION = 1

try:
    fg = fs.get_feature_group(FG_NAME, version=FG_VERSION)
    print(f"Found existing feature group: {FG_NAME} v{FG_VERSION}")
except Exception as e:
    print(f"Creating feature group: {FG_NAME}")
    fg = fs.create_feature_group(
        name=FG_NAME,
        version=FG_VERSION,
        primary_key=["entity_id"],
        event_time="event_time",
        description="Daily feature observations for drift investigation",
        statistics_config={
            "enabled": True,
            "histograms": True,
            "correlations": False,
            "exact_uniqueness": False,
        },
    )
    fg.save(df)
    print("Feature group created and data inserted")

# Explore monitoring capabilities
print("\n--- Exploring platform statistics/monitoring ---")
print("Feature group methods:", [m for m in dir(fg) if not m.startswith('_')])

# Get statistics from the platform
try:
    stats = fg.get_statistics()
    print(f"\nStatistics retrieved: {type(stats)}")
    if stats is not None:
        print(f"Stats content: {stats}")
except Exception as e:
    print(f"get_statistics error: {e}")

# Try feature monitoring
try:
    fm = fg.get_feature_monitoring_configs()
    print(f"\nFeature monitoring configs: {fm}")
except Exception as e:
    print(f"Feature monitoring error: {e}")

# Explore feature monitoring API
try:
    import hsfs.feature_monitoring as fm_module
    print("\nFeature monitoring module:", dir(fm_module))
except Exception as e:
    print(f"Import error: {e}")

# --- Use platform statistics to compute per-date windows ---
# Try to get statistics for specific time windows
print("\n--- Attempting windowed statistics ---")

# First, let's try to compute statistics for early period vs late period
# Split data at the midpoint to look for drift
midpoint = len(dates) // 2
early_dates = dates[:midpoint]
late_dates = dates[midpoint:]

early_df = df[df["event_time"].dt.date.isin(early_dates)]
late_df = df[df["event_time"].dt.date.isin(late_dates)]

print(f"Early period: {early_dates[0]} to {early_dates[-1]} ({len(early_dates)} days)")
print(f"Late period: {late_dates[0]} to {late_dates[-1]} ({len(late_dates)} days)")

# Compute basic stats per period (pandas is allowed for basic computation)
print("\nFeature means by period:")
for f in features:
    early_mean = early_df[f].mean()
    late_mean = late_df[f].mean()
    early_std = early_df[f].std()
    late_std = late_df[f].std()
    z_score = abs(late_mean - early_mean) / (early_std / (len(early_df)**0.5))
    print(f"  {f}: early_mean={early_mean:.4f}, late_mean={late_mean:.4f}, z_score={z_score:.2f}")

# More detailed analysis: compute daily means and look for changepoints
print("\n--- Daily mean analysis (sliding window changepoint detection) ---")
window = 14  # 2-week window

daily_means = {}
for f in features:
    daily_means[f] = []
    for d in dates:
        day_df = df[df["event_time"].dt.date == d]
        daily_means[f].append(day_df[f].mean())

best_feature = None
best_date = None
best_score = 0

feature_scores = {}
for f in features:
    means = daily_means[f]
    feat_best_score = 0
    feat_best_date = None
    for i in range(window, len(dates) - window):
        before = means[max(0, i-window):i]
        after = means[i:i+window]
        before_avg = sum(before) / len(before)
        after_avg = sum(after) / len(after)
        before_var = sum((x - before_avg)**2 for x in before) / len(before)
        before_std = before_var ** 0.5

        if before_std > 0.01:
            score = abs(after_avg - before_avg) / before_std
        else:
            score = 0

        if score > feat_best_score:
            feat_best_score = score
            feat_best_date = str(dates[i])

    feature_scores[f] = (feat_best_score, feat_best_date)
    if feat_best_score > best_score:
        best_score = feat_best_score
        best_feature = f
        best_date = feat_best_date

print("\nPer-feature drift scores:")
for f in features:
    score, onset = feature_scores[f]
    marker = " <<< DRIFTED" if f == best_feature else ""
    print(f"  {f}: score={score:.4f}, onset={onset}{marker}")

print(f"\nConclusion: feature={best_feature}, onset={best_date}, score={best_score:.4f}")

# Validate by showing the means around the onset for the drifted feature
if best_feature and best_date:
    print(f"\nDaily means for {best_feature} around onset {best_date}:")
    onset_idx = dates.index(pd.to_datetime(best_date).date())
    start_idx = max(0, onset_idx - 5)
    end_idx = min(len(dates), onset_idx + 10)
    for i in range(start_idx, end_idx):
        d = dates[i]
        m = daily_means[best_feature][i]
        marker = " <<< DRIFT ONSET" if i == onset_idx else ""
        print(f"  {d}: {m:.4f}{marker}")

# Write results
os.makedirs("submission", exist_ok=True)
result = {"feature": best_feature, "onset": best_date}
with open("submission/answers.json", "w") as fp:
    json.dump(result, fp)
print(f"\nAnswer written to submission/answers.json: {result}")
