import hopsworks
import pandas as pd
import json
import os
from datetime import date, timedelta


def main():
    project = hopsworks.login()
    fs = project.get_feature_store()
    dataset_api = project.get_dataset_api()

    # Download the prediction log from HopsFS
    dataset_api.download("Resources/prediction_log.csv", "/tmp/prediction_log.csv", overwrite=True)

    df = pd.read_csv("/tmp/prediction_log.csv")
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").reset_index(drop=True)

    print(f"Loaded {len(df)} predictions from {df['ts'].min()} to {df['ts'].max()}")

    # Create feature group with event_time for platform-side monitoring
    from hsfs.statistics_config import StatisticsConfig

    fg = fs.get_or_create_feature_group(
        name="prediction_log",
        version=1,
        primary_key=["ts"],
        event_time="ts",
        description="Prediction monitoring log for distribution shift detection",
        statistics_config=StatisticsConfig(
            enabled=True,
            histograms=True,
            correlations=False,
        ),
    )

    fg.insert(df, write_options={"wait_for_job": True})
    print("Data inserted into feature group.")

    # -----------------------------------------------------------------------
    # Attempt feature monitoring: compare weekly windows against baseline
    # Today is June 20, 2026; data spans Jan 1 – Mar 31, 2026
    # -----------------------------------------------------------------------
    TODAY = date(2026, 6, 20)
    DATA_START = date(2026, 1, 1)
    DATA_END = date(2026, 3, 31)

    # Baseline: first 14 days (Jan 1-14)
    baseline_end = DATA_START + timedelta(days=14)
    # time_offset for reference = days from today to the START of the reference window
    ref_offset_days = (TODAY - DATA_START).days    # 170 days
    ref_window_length_days = 14

    onset_date = None
    fm_worked = False

    scan = baseline_end  # start scanning from Jan 15
    while scan <= DATA_END - timedelta(days=7):
        det_offset_days = (TODAY - scan).days
        config_name = f"pred_drift_{scan.strftime('%Y%m%d')}"

        try:
            # Clean up any previous config with the same name
            try:
                existing = fg.get_feature_monitoring_configs(name=config_name)
                items = existing if isinstance(existing, list) else [existing]
                for item in items:
                    if item is not None:
                        item.delete()
            except Exception:
                pass

            config = (
                fg.create_feature_monitoring(
                    name=config_name,
                    feature_name="prediction",
                    cron_expression="0 0 12 ? * * *",
                )
                .with_detection_window(
                    time_offset=f"{det_offset_days}d",
                    window_length="7d",
                )
                .with_reference_window(
                    time_offset=f"{ref_offset_days}d",
                    window_length=f"{ref_window_length_days}d",
                )
                .compare_on(
                    metric="mean",
                    threshold=0.5,
                    relative=True,
                    strict=False,
                )
                .save()
            )

            config.run_job()
            fm_worked = True

            history = config.get_history(with_statistics=True)
            items = history if isinstance(history, list) else [history]
            for item in items:
                if item is not None and getattr(item, "shift_detected", False):
                    onset_date = scan.strftime("%Y-%m-%d")
                    print(f"Feature monitoring: shift detected at week starting {onset_date}")
                    break

            if onset_date:
                break

        except Exception as e:
            err_str = str(e)
            print(f"Feature monitoring for {scan}: {err_str}")
            if "270234" in err_str or "not enabled" in err_str.lower():
                print("Feature monitoring service not enabled; switching to statistics.")
                break

        scan += timedelta(days=7)

    # -----------------------------------------------------------------------
    # Fallback: compute statistics within this job (runs on platform compute)
    # Use CUSUM change-point detection on daily prediction means
    # -----------------------------------------------------------------------
    if onset_date is None:
        print("Computing shift onset via CUSUM on daily means...")

        df["date"] = df["ts"].dt.date
        daily = df.groupby("date")["prediction"].agg(["mean", "std", "count"]).reset_index()
        daily = daily.sort_values("date").reset_index(drop=True)

        dates = daily["date"].tolist()
        means = daily["mean"].tolist()

        # Baseline: first 14 days
        baseline_vals = means[:14]
        baseline_mean = sum(baseline_vals) / len(baseline_vals)
        baseline_variance = sum((v - baseline_mean) ** 2 for v in baseline_vals) / len(baseline_vals)
        baseline_std = baseline_variance ** 0.5

        print(f"Baseline mean: {baseline_mean:.4f}, std: {baseline_std:.4f}")

        # CUSUM parameters
        delta = baseline_std        # expected shift size (1 sigma)
        k = delta / 2.0             # allowance
        h = 4.0 * baseline_std      # decision threshold

        cusum_pos = 0.0
        cusum_neg = 0.0

        for i in range(14, len(dates)):
            m = means[i]
            cusum_pos = max(0.0, cusum_pos + (m - baseline_mean) - k)
            cusum_neg = max(0.0, cusum_neg - (m - baseline_mean) - k)

            if cusum_pos >= h or cusum_neg >= h:
                d = dates[i]
                onset_date = d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
                print(f"CUSUM detected shift at {onset_date} (cusum_pos={cusum_pos:.3f}, cusum_neg={cusum_neg:.3f})")
                break

        if onset_date is None:
            # Try lower threshold
            cusum_pos = 0.0
            for i in range(14, len(dates)):
                m = means[i]
                cusum_pos = max(0.0, cusum_pos + (m - baseline_mean) - k / 2)
                if cusum_pos >= h / 2:
                    d = dates[i]
                    onset_date = d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
                    print(f"CUSUM (low threshold) detected shift at {onset_date}")
                    break

    result = {"onset": onset_date}
    print(f"FINAL RESULT: {json.dumps(result)}")

    # Save result to HopsFS so it can be read back
    with open("/tmp/answers.json", "w") as f:
        json.dump(result, f)
    dataset_api.upload("/tmp/answers.json", "Resources", overwrite=True)
    print("Result uploaded to HopsFS Resources/answers.json")


if __name__ == "__main__":
    main()
