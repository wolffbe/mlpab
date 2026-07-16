"""
Drift detection job: reads feature CSV from HopsFS, inserts into feature group,
computes per-day statistics, identifies drifted feature and onset date.
"""
import hopsworks
import os
import json
import math


def compute_stats(values):
    n = len(values)
    if n == 0:
        return None, None
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    return mean, math.sqrt(var)


def main():
    project = hopsworks.login()
    fs = project.get_feature_store()
    dataset_api = project.get_dataset_api()

    # Download CSV from HopsFS
    dataset_api.download("Resources/features.csv", "/tmp/")
    print("Downloaded features.csv from HopsFS")

    import pandas as pd

    df = pd.read_csv("/tmp/features.csv", parse_dates=["event_time"])
    df["event_date"] = df["event_time"].dt.date
    df = df.sort_values("event_time").reset_index(drop=True)

    features = ["f1", "f2", "f3", "f4", "f5", "f6"]
    dates = sorted(df["event_date"].unique())

    print(f"Loaded {len(df)} rows, {len(dates)} dates: {dates[0]} to {dates[-1]}")

    # Insert into feature group (uses platform statistics engine)
    fg = fs.get_feature_group("drift_detection", version=1)
    df_insert = df.drop(columns=["event_date"])
    print("Inserting data into feature group...")
    fg.insert(df_insert, wait=False)
    print("Insert triggered")

    # Compute per-day statistics for drift detection
    daily_means = {f: [] for f in features}
    daily_stds = {f: [] for f in features}
    date_strs = []

    for d in dates:
        day_df = df[df["event_date"] == d]
        date_strs.append(str(d))
        for f in features:
            vals = day_df[f].dropna().tolist()
            mean, std = compute_stats(vals)
            daily_means[f].append(mean if mean is not None else 0.0)
            daily_stds[f].append(std if std is not None else 0.0)

    n_dates = len(dates)
    n_baseline = 30  # First 30 days as baseline

    # Baseline: mean and std of daily means
    baseline_stats = {}
    for f in features:
        mu, sig = compute_stats(daily_means[f][:n_baseline])
        baseline_stats[f] = {"mu": mu, "sig": sig}

    print("\nBaseline statistics (mean of daily means, std of daily means):")
    for f in features:
        b = baseline_stats[f]
        print(f"  {f}: mu={b['mu']:.4f}, sig={b['sig']:.4f}")

    # Post-baseline z-scores per feature per day
    print("\nPost-baseline daily z-scores:")
    feature_max_z = {f: 0.0 for f in features}
    feature_z_series = {f: [] for f in features}

    for i in range(n_baseline, n_dates):
        for f in features:
            b = baseline_stats[f]
            if b["sig"] and b["sig"] > 1e-8:
                z = abs(daily_means[f][i] - b["mu"]) / b["sig"]
            else:
                z = 0.0
            feature_z_series[f].append((date_strs[i], daily_means[f][i], z))
            if z > feature_max_z[f]:
                feature_max_z[f] = z

    print("\nMax z-scores per feature (post-baseline):")
    for f, z in sorted(feature_max_z.items(), key=lambda x: -x[1]):
        print(f"  {f}: max_z={z:.3f}")

    # Drifted feature = highest max z-score
    drifted_feature = max(feature_max_z, key=feature_max_z.get)
    print(f"\nDrifted feature: {drifted_feature}")

    # Print full z-score series for drifted feature
    print(f"\nDaily z-scores for {drifted_feature}:")
    for date_s, mean_v, z in feature_z_series[drifted_feature]:
        marker = " <-- DRIFT" if z > 2.0 else ""
        print(f"  {date_s}: mean={mean_v:.4f}, z={z:.3f}{marker}")

    # Find onset: first day where individual z-score > 2.0 after baseline
    threshold = 2.0
    onset = None
    for date_s, mean_v, z in feature_z_series[drifted_feature]:
        if z > threshold:
            onset = date_s
            break

    # Confirm with 3-day consecutive check (robustness)
    if onset is not None:
        # Find first 3 consecutive days all above threshold/2
        series = feature_z_series[drifted_feature]
        onset_confirmed = None
        for i in range(len(series) - 2):
            if (series[i][2] > threshold / 2
                    and series[i + 1][2] > threshold / 2
                    and series[i + 2][2] > threshold / 2):
                onset_confirmed = series[i][0]
                break
        if onset_confirmed:
            onset = onset_confirmed
            print(f"Onset confirmed by 3-consecutive-day check: {onset}")

    print(f"\n=== ANSWER ===")
    print(f"Feature: {drifted_feature}")
    print(f"Onset: {onset}")

    answer = {"feature": drifted_feature, "onset": onset}
    print(f"\n===FINAL_ANSWER=== {json.dumps(answer)}")

    # Save to /tmp and upload to HopsFS
    with open("/tmp/drift_answer.json", "w") as f_out:
        json.dump(answer, f_out)

    dataset_api.upload("/tmp/drift_answer.json", "Resources/drift_answer.json", overwrite=True)
    print("Uploaded answer to HopsFS Resources/drift_answer.json")


if __name__ == "__main__":
    main()
