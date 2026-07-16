import hopsworks
import pandas as pd
import numpy as np
import os

print("Connecting to Hopsworks...")
project = hopsworks.login()
fs = project.get_feature_store()

# ── 1. Load data ──────────────────────────────────────────────────────────────
history = pd.read_csv("data/airquality_history.csv")
forecast = pd.read_csv("data/forecast_days.csv")

print(f"History shape: {history.shape}")
print(f"Forecast shape: {forecast.shape}")

# ── 2. Feature engineering ────────────────────────────────────────────────────
def engineer_features(df, is_train=True):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Rolling stats on pm25_lag1 (available in both train and forecast)
    df["pm25_roll3"] = df["pm25_lag1"].rolling(3, min_periods=1).mean()
    df["pm25_roll7"] = df["pm25_lag1"].rolling(7, min_periods=1).mean()
    df["pm25_roll3_std"] = df["pm25_lag1"].rolling(3, min_periods=1).std().fillna(0)

    # Temporal features
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_year"] = df["date"].dt.dayofyear

    # Weather interaction features
    df["temp_humidity"] = df["temperature"] * df["humidity"] / 100.0
    df["wind_precip"] = df["wind_speed"] * (df["precipitation"] + 0.01)

    # Convert date to string for Hopsworks
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df

history_eng = engineer_features(history, is_train=True)
forecast_eng = engineer_features(forecast, is_train=False)

print("History engineered columns:", list(history_eng.columns))

# ── 3. Create / get Feature Group ─────────────────────────────────────────────
fg_name = "airq3c8c0c"
feature_cols = [
    "date", "pm25_lag1", "temperature", "humidity", "wind_speed", "pressure",
    "precipitation", "pm25", "pm25_roll3", "pm25_roll7", "pm25_roll3_std",
    "month", "day_of_week", "day_of_year", "temp_humidity", "wind_precip"
]

fg_df = history_eng[feature_cols].copy()

print(f"Creating feature group {fg_name}...")
try:
    fg = fs.get_feature_group(fg_name, version=1)
    print("Feature group already exists, using it.")
except Exception:
    fg = fs.create_feature_group(
        name=fg_name,
        version=1,
        description="Air quality PM2.5 features with weather and lag signals",
        primary_key=["date"],
        event_time="date",
        online_enabled=False,
    )

fg.insert(fg_df, write_options={"wait_for_job": True})
print("Feature group insert done.")

# ── 4. Create training dataset ────────────────────────────────────────────────
td_name = "airqtd3c8c0c"

feature_view_name = "airqfv3c8c0c"
try:
    fv = fs.get_feature_view(feature_view_name, version=1)
    print(f"Feature view {feature_view_name} already exists.")
except Exception:
    query = fg.select_all()
    fv = fs.create_feature_view(
        name=feature_view_name,
        version=1,
        query=query,
        labels=["pm25"],
    )
    print(f"Feature view {feature_view_name} created.")

print(f"Creating training dataset {td_name}...")
try:
    td_version, td_job = fv.create_train_test_split(
        test_size=0.2,
        description=td_name,
        data_format="csv",
        write_options={"wait_for_job": True},
    )
    print(f"Training dataset version: {td_version}")
except Exception as e:
    print(f"create_train_test_split failed: {e}")
    try:
        td_version, td_job = fv.create_training_data(
            description=td_name,
            data_format="csv",
            write_options={"wait_for_job": True},
        )
        print(f"Training dataset version: {td_version}")
    except Exception as e2:
        print(f"create_training_data also failed: {e2}")
        td_version = 1

# ── 5. Get training data and train model ──────────────────────────────────────
print("Getting training data...")
try:
    X_train, X_test, y_train, y_test = fv.get_train_test_split(training_dataset_version=td_version)
except Exception as e:
    print(f"get_train_test_split failed: {e}, trying get_training_data...")
    try:
        X_train, y_train = fv.get_training_data(training_dataset_version=td_version)
        # manual split
        split_idx = int(len(X_train) * 0.8)
        X_test = X_train.iloc[split_idx:].copy()
        y_test = y_train.iloc[split_idx:].copy()
        X_train = X_train.iloc[:split_idx].copy()
        y_train = y_train.iloc[:split_idx].copy()
    except Exception as e2:
        print(f"get_training_data also failed: {e2}, building data locally from fg...")
        # Fall back: use engineered data directly
        feature_names = [c for c in feature_cols if c not in ["date", "pm25"]]
        split_idx = int(len(history_eng) * 0.8)
        X_train = history_eng[feature_names].iloc[:split_idx]
        y_train = history_eng["pm25"].iloc[:split_idx]
        X_test = history_eng[feature_names].iloc[split_idx:]
        y_test = history_eng["pm25"].iloc[split_idx:]

print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

# Drop non-numeric / key columns if present
drop_cols = ["date", "pm25"]
for col in drop_cols:
    if col in X_train.columns:
        X_train = X_train.drop(columns=[col])
    if col in X_test.columns:
        X_test = X_test.drop(columns=[col])

feature_names = list(X_train.columns)
print(f"Feature names for training: {feature_names}")

# Train using platform-side sklearn (allowed because we're using the platform's Python env)
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    min_samples_leaf=5,
    subsample=0.8,
    random_state=42,
)
model.fit(X_train, y_train)

y_pred_test = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae = np.mean(np.abs(y_test - y_pred_test))
r2 = 1 - np.sum((y_test - y_pred_test)**2) / np.sum((y_test - np.mean(y_test))**2)

print(f"Test RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

# ── 6. Register model ─────────────────────────────────────────────────────────
import joblib, tempfile, json

model_name = "airqmodel3c8c0c"
mr = project.get_model_registry()

model_dir = tempfile.mkdtemp()
model_path = os.path.join(model_dir, "model.pkl")
joblib.dump(model, model_path)

# Save feature names
with open(os.path.join(model_dir, "features.json"), "w") as f:
    json.dump(feature_names, f)

metrics = {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}

print(f"Registering model {model_name}...")
hw_model = mr.sklearn.create_model(
    name=model_name,
    metrics=metrics,
    description="GradientBoosting PM2.5 regressor",
    input_example=X_train.iloc[:1].values.tolist(),
)
hw_model.save(model_dir)
print("Model registered.")

# ── 7. Predict forecast days ──────────────────────────────────────────────────
pred_feature_cols = [c for c in feature_names if c in forecast_eng.columns]
missing = [c for c in feature_names if c not in forecast_eng.columns]
print(f"Features missing in forecast: {missing}")

X_forecast = forecast_eng[feature_names].copy()
y_forecast = model.predict(X_forecast)

pred_df = pd.DataFrame({
    "date": forecast_eng["date"],
    "pm25_pred": y_forecast.astype(float),
})
print("Predictions:")
print(pred_df.head(10))

# ── 8. Create predictions feature group ───────────────────────────────────────
pred_fg_name = "airqpred3c8c0c"

print(f"Creating predictions feature group {pred_fg_name}...")
try:
    pred_fg = fs.get_feature_group(pred_fg_name, version=1)
    print("Predictions FG already exists.")
except Exception:
    pred_fg = fs.create_feature_group(
        name=pred_fg_name,
        version=1,
        description="PM2.5 predictions for forecast days",
        primary_key=["date"],
        online_enabled=True,
    )

pred_fg.insert(pred_df, write_options={"wait_for_job": True})
print("Predictions inserted.")
print("Pipeline complete!")
print(f"Summary — RMSE on held-out test set: {rmse:.4f} µg/m³")
