"""Step 1: Feature engineering + ingestion into Hopsworks (no ML libraries)."""
import hopsworks
import pandas as pd
import numpy as np

print("Connecting to Hopsworks...")
project = hopsworks.login()
fs = project.get_feature_store()

# ── Load data ─────────────────────────────────────────────────────────────────
history = pd.read_csv("data/airquality_history.csv")
forecast = pd.read_csv("data/forecast_days.csv")

print(f"History shape: {history.shape}")
print(f"Forecast shape: {forecast.shape}")

def engineer_features(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df["pm25_roll3"] = df["pm25_lag1"].rolling(3, min_periods=1).mean()
    df["pm25_roll7"] = df["pm25_lag1"].rolling(7, min_periods=1).mean()
    df["pm25_roll3_std"] = df["pm25_lag1"].rolling(3, min_periods=1).std().fillna(0)

    df["month"] = df["date"].dt.month.astype(float)
    df["day_of_week"] = df["date"].dt.dayofweek.astype(float)
    df["day_of_year"] = df["date"].dt.dayofyear.astype(float)

    df["temp_humidity"] = df["temperature"] * df["humidity"] / 100.0
    df["wind_precip"] = df["wind_speed"] * (df["precipitation"] + 0.01)

    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df

history_eng = engineer_features(history)
forecast_eng = engineer_features(forecast)

# ── Feature Group for history ─────────────────────────────────────────────────
fg_name = "airq3c8c0c"
feature_cols_hist = [
    "date", "pm25_lag1", "temperature", "humidity", "wind_speed", "pressure",
    "precipitation", "pm25", "pm25_roll3", "pm25_roll7", "pm25_roll3_std",
    "month", "day_of_week", "day_of_year", "temp_humidity", "wind_precip"
]
fg_df = history_eng[feature_cols_hist].copy()

print(f"Creating/getting feature group {fg_name}...")
fg = fs.get_feature_group(fg_name, version=1)
if fg is None:
    fg = fs.create_feature_group(
        name=fg_name,
        version=1,
        description="Air quality PM2.5 features with weather and lag signals",
        primary_key=["date"],
        online_enabled=False,
    )
    print("Feature group created.")
else:
    print("Feature group exists.")

print("Inserting history data...")
fg.insert(fg_df, write_options={"wait_for_job": True})
print("History data inserted.")

# ── Feature View ──────────────────────────────────────────────────────────────
fv_name = "airqfv3c8c0c"
fv = fs.get_feature_view(fv_name, version=1)
if fv is None:
    query = fg.select_all()
    fv = fs.create_feature_view(
        name=fv_name,
        version=1,
        query=query,
        labels=["pm25"],
    )
    print(f"Feature view {fv_name} created.")
else:
    print(f"Feature view {fv_name} exists.")

# ── Training dataset ──────────────────────────────────────────────────────────
print("Creating training dataset...")
try:
    td_version, td_job = fv.create_train_test_split(
        test_size=0.2,
        description="airqtd3c8c0c",
        data_format="csv",
        write_options={"wait_for_job": True},
    )
    print(f"Training dataset version: {td_version}")
except Exception as e:
    print(f"create_train_test_split failed: {e}")
    try:
        td_version, td_job = fv.create_training_data(
            description="airqtd3c8c0c",
            data_format="csv",
            write_options={"wait_for_job": True},
        )
        print(f"Training dataset version: {td_version}")
    except Exception as e2:
        print(f"create_training_data also failed: {e2}")
        td_version = 1

print(f"TD version: {td_version}")

# ── Save forecast data to predictions FG placeholder ─────────────────────────
# We store forecast features for later use by the training job
pred_fg_name = "airqpred3c8c0c"
# Build forecast with placeholder pm25_pred=0.0 for now
feature_cols_fc = [
    "date", "pm25_lag1", "temperature", "humidity", "wind_speed", "pressure",
    "precipitation", "pm25_roll3", "pm25_roll7", "pm25_roll3_std",
    "month", "day_of_week", "day_of_year", "temp_humidity", "wind_precip"
]
forecast_upload = forecast_eng[feature_cols_fc].copy()
forecast_upload["pm25_pred"] = 0.0

print(f"Creating/getting predictions feature group {pred_fg_name}...")
pred_fg = fs.get_feature_group(pred_fg_name, version=1)
if pred_fg is None:
    pred_fg = fs.create_feature_group(
        name=pred_fg_name,
        version=1,
        description="PM2.5 predictions for forecast days",
        primary_key=["date"],
        online_enabled=True,
    )
    print("Predictions FG created.")
else:
    print("Predictions FG exists.")

pred_fg.insert(forecast_upload, write_options={"wait_for_job": True})
print("Forecast placeholder inserted.")

print(f"\nIngestion done. td_version={td_version}, fv={fv_name}")
print("Now run: python train_job.py")
