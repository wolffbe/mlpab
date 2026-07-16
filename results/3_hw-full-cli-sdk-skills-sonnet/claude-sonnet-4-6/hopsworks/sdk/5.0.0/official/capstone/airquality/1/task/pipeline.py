import hopsworks
import pandas as pd
import numpy as np
import os
import json
import joblib

# ── 1. Connect ────────────────────────────────────────────────────────────────
project = hopsworks.login()
fs = project.get_feature_store()

# ── 2. Load data ──────────────────────────────────────────────────────────────
history_df = pd.read_csv("data/airquality_history.csv", parse_dates=["date"])
forecast_df = pd.read_csv("data/forecast_days.csv",      parse_dates=["date"])

def engineer(df):
    df = df.copy()
    df["month"]       = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    # rolling features from lag1 as proxy (same value available at forecast time)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df

history_df  = engineer(history_df)
forecast_df = engineer(forecast_df)

print("History shape:", history_df.shape)
print("Forecast shape:", forecast_df.shape)
print(history_df.dtypes)

# ── 3. Feature group ──────────────────────────────────────────────────────────
fg = fs.get_or_create_feature_group(
    name="airq3c8c0c",
    version=1,
    primary_key=["date"],
    description="Air quality + weather features",
    online_enabled=True,
)
fg.insert(history_df, write_options={"wait_for_job": True})
print("Feature group insert done.")

# ── 4. Feature view / training dataset ───────────────────────────────────────
fv = fs.get_or_create_feature_view(
    name="airqtd3c8c0c",
    version=1,
    query=fg.select_all(),
    labels=["pm25"],
)
print("Feature view created.")

# Get training data
X_train, X_test, y_train, y_test = fv.train_test_split(test_size=0.2)
print("Training set:", X_train.shape, "Test set:", X_test.shape)

# ── 5. Train model ────────────────────────────────────────────────────────────
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

model = GradientBoostingRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    random_state=42,
)
model.fit(X_train, y_train)

y_pred_test = model.predict(X_test)
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
mae  = float(mean_absolute_error(y_test, y_pred_test))
print(f"Test RMSE={rmse:.4f}  MAE={mae:.4f}")

# ── 6. Register model ─────────────────────────────────────────────────────────
model_dir = "./airq_model_dir"
os.makedirs(model_dir, exist_ok=True)
joblib.dump(model, os.path.join(model_dir, "model.pkl"))

# Save feature column order for reproducible inference
feature_cols = list(X_train.columns)
with open(os.path.join(model_dir, "feature_cols.json"), "w") as f:
    json.dump(feature_cols, f)

mr = project.get_model_registry()
hw_model = mr.sklearn.create_model(
    name="airqmodel3c8c0c",
    version=1,
    metrics={"rmse": rmse, "mae": mae},
    description="PM2.5 GradientBoosting regressor",
    input_example=X_train.head(3),
    feature_view=fv,
)
hw_model.save(model_dir)
print("Model registered:", hw_model.name, "v", hw_model.version)

# ── 7. Predict forecast rows ──────────────────────────────────────────────────
X_forecast = forecast_df[feature_cols]
pm25_pred = model.predict(X_forecast)

pred_df = pd.DataFrame({
    "date":      forecast_df["date"],
    "pm25_pred": pm25_pred.astype(float),
})
print("Predictions sample:")
print(pred_df.head())

# ── 8. Predictions feature group (online-enabled for low-latency lookup) ──────
pred_fg = fs.get_or_create_feature_group(
    name="airqpred3c8c0c",
    version=1,
    primary_key=["date"],
    description="PM2.5 forecast predictions",
    online_enabled=True,
)
pred_fg.insert(pred_df, write_options={"wait_for_job": True})
print("Predictions inserted into airqpred3c8c0c.")
print("Done.")
