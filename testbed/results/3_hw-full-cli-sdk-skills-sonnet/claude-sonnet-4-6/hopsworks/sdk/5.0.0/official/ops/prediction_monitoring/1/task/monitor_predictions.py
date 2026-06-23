"""
Log predictions into Hopsworks and detect distribution shift using
feature monitoring + platform SQL statistics.
"""
import hopsworks
import pandas as pd
import json
import os
from hsfs.statistics_config import StatisticsConfig

# ── 1. Connect ──────────────────────────────────────────────────────────────
project = hopsworks.login()
fs = project.get_feature_store()
print(f"Connected: project={project.name}, offline_store={fs.offline_featurestore_name}")

# ── 2. Load data ─────────────────────────────────────────────────────────────
df = pd.read_csv("data/prediction_log.csv")
df["ts"] = pd.to_datetime(df["ts"])
df["date"] = df["ts"].dt.strftime("%Y-%m-%d")
df["id"] = range(len(df))
print(f"Loaded {len(df)} predictions spanning {df['date'].min()} → {df['date'].max()}")
print(f"Prediction range: {df['prediction'].min():.3f} – {df['prediction'].max():.3f}")

# ── 3. Create / get feature group ─────────────────────────────────────────
fg = fs.get_or_create_feature_group(
    name="prediction_log",
    version=1,
    primary_key=["id"],
    description="Logged predictions for drift monitoring",
    statistics_config=StatisticsConfig(
        enabled=True,
        histograms=True,
        correlations=False,
        exact_uniqueness=False,
    ),
    event_time="ts",
)
print(f"Feature group: {fg.name} v{fg.version}")

# ── 4. Insert all predictions (one shot) ──────────────────────────────────
print("Inserting predictions...")
fg.insert(df[["id", "ts", "date", "prediction"]])
print("Insert complete.")
# Try to get statistics from insert (computed automatically)
try:
    stats = fg.get_statistics()
    if stats:
        for fds in stats.feature_descriptive_statistics:
            if fds.feature_name == "prediction":
                print(f"Overall prediction stats: mean={fds.mean:.4f}, std={fds.stddev:.4f}")
except Exception as e:
    print(f"Stats retrieval note: {e}")

# ── 5. Platform SQL: compute daily mean / std ────────────────────────────
# Table name in the offline store follows the pattern: {fg_name}_{fg_version}
table_name = f"`{fg.name}_{fg.version}`"
print(f"\nRunning platform SQL on {table_name} …")
try:
    daily_df = fs.sql(
        f"SELECT date, "
        f"AVG(prediction) AS mean_pred, "
        f"STDDEV(prediction) AS std_pred, "
        f"MIN(prediction) AS min_pred, "
        f"MAX(prediction) AS max_pred, "
        f"COUNT(*) AS cnt "
        f"FROM {table_name} "
        f"GROUP BY date "
        f"ORDER BY date",
    )
    print(f"Daily stats retrieved: {len(daily_df)} rows")
    print(daily_df.to_string())
except Exception as e:
    print(f"SQL error: {e}")
    daily_df = None

# ── 6. Feature monitoring setup ────────────────────────────────────────────
# Set reference = mean of first 7 days
if daily_df is not None and len(daily_df) > 7:
    ref_mean = float(daily_df.head(7)["mean_pred"].mean())
    print(f"\nReference mean (first 7 days): {ref_mean:.4f}")
    try:
        existing = fg.get_feature_monitoring_configs(name="pred_drift_v1")
        if existing:
            print("Monitoring config already exists, deleting...")
            existing[0].delete()
    except Exception:
        pass

    try:
        config = fg.create_feature_monitoring(
            name="pred_drift_v1",
            feature_name="prediction",
            description="Detect when prediction mean shifts from baseline",
        ).with_detection_window(
            time_offset="1d",
            window_length="1d",
        ).with_reference_value(
            value=ref_mean,
        ).compare_on(
            metric="mean",
            threshold=1.0,
            relative=False,
            strict=False,
        ).save()
        print(f"Feature monitoring config created: {config}")
        config.run_job()
        history = config.get_history(with_statistics=True)
        print(f"Monitoring history: {history}")
    except Exception as e:
        print(f"Feature monitoring error (cluster service may not be enabled): {e}")

# ── 7. Detect shift from daily SQL stats ─────────────────────────────────
if daily_df is not None and len(daily_df) > 0:
    # Compute rolling 7-day mean to smooth noise
    daily_df = daily_df.sort_values("date").reset_index(drop=True)
    daily_df["rolling_mean"] = daily_df["mean_pred"].rolling(window=7, center=False).mean()

    # Baseline: mean of first 14 days
    baseline_mean = float(daily_df.head(14)["mean_pred"].mean())
    baseline_std = float(daily_df.head(14)["mean_pred"].std())
    threshold = baseline_mean + 2 * baseline_std
    print(f"\nBaseline (first 14 days): mean={baseline_mean:.4f}, std={baseline_std:.4f}")
    print(f"Shift threshold: {threshold:.4f}")

    # Find first day where rolling mean persistently exceeds threshold
    onset_date = None
    # Use 7-day rolling mean to find persistent shift
    for i, row in daily_df.iterrows():
        if i < 7:
            continue
        if row["rolling_mean"] is not None and float(row["rolling_mean"]) > threshold:
            # Confirm with subsequent days
            future = daily_df.iloc[i:i+7]["rolling_mean"].dropna()
            if len(future) >= 5 and all(v > threshold for v in future):
                onset_date = row["date"]
                break

    # If rolling mean approach didn't work, try simple threshold on daily mean
    if onset_date is None:
        for i, row in daily_df.iterrows():
            if float(row["mean_pred"]) > threshold:
                # Check that this isn't a one-off spike - verify 5 of next 7 days also exceed
                future = daily_df.iloc[i:i+7]["mean_pred"]
                if len(future) >= 5 and sum(v > threshold for v in future) >= 5:
                    onset_date = row["date"]
                    break

    print(f"\nDetected shift onset: {onset_date}")

    # Save answer
    os.makedirs("submission", exist_ok=True)
    with open("submission/answers.json", "w") as f:
        json.dump({"onset": onset_date}, f)
    print(f"Answer written: submission/answers.json → onset={onset_date}")
else:
    print("ERROR: Could not get daily statistics from platform SQL")
