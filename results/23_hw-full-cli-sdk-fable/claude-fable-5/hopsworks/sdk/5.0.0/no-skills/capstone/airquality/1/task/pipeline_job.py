"""Full FTI pipeline for PM2.5 forecasting. Runs as a Hopsworks PYTHON job on the cluster."""

import os

import hopsworks
import numpy as np
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()

# the job script lives in /hopsfs/Resources/airq754fa9 alongside the uploaded CSVs
script_dir = os.path.dirname(os.path.abspath(__file__))
hist_path = os.path.join(script_dir, "airquality_history.csv")
fc_path = os.path.join(script_dir, "forecast_days.csv")
if not os.path.exists(hist_path):
    dataset_api = project.get_dataset_api()
    hist_path = dataset_api.download(f"/Projects/{project.name}/Resources/airq754fa9/airquality_history.csv", overwrite=True)
    fc_path = dataset_api.download(f"/Projects/{project.name}/Resources/airq754fa9/forecast_days.csv", overwrite=True)

hist = pd.read_csv(hist_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
fc = pd.read_csv(fc_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
print("history rows:", len(hist), "forecast rows:", len(fc), flush=True)

WEATHER = ["temperature", "humidity", "wind_speed", "pressure", "precipitation"]


def engineer(df):
    out = df.copy()
    doy = out["date"].dt.dayofyear.astype(float)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    out["month"] = out["date"].dt.month.astype("int64")
    out["temp_x_wind"] = out["temperature"] * out["wind_speed"]
    out["humidity_x_precip"] = out["humidity"] * out["precipitation"]
    out["lag1_x_lowwind"] = out["pm25_lag1"] * (out["wind_speed"] < 10.0).astype(float)
    return out


hist_e = engineer(hist)
# lag/rolling air-quality signals over the contiguous daily history
hist_e["pm25_lag2"] = hist_e["pm25"].shift(2).fillna(hist_e["pm25_lag1"])
hist_e["pm25_roll3"] = hist_e["pm25"].shift(1).rolling(3, min_periods=1).mean()
hist_e["pm25_roll7"] = hist_e["pm25"].shift(1).rolling(7, min_periods=1).mean()

# ---------------- 1. Feature group airq754fa9 ----------------
fg_df = hist_e.copy()
fg_df["date"] = fg_df["date"].dt.strftime("%Y-%m-%d")
fg = fs.get_or_create_feature_group(
    name="airq754fa9",
    version=1,
    primary_key=["date"],
    description="Engineered daily air-quality features: weather, seasonal encodings, PM2.5 lag/rolling signals, target pm25",
    online_enabled=False,
)
fg.insert(fg_df, write_options={"wait_for_job": True})
print("feature group airq754fa9 inserted:", len(fg_df), "rows", flush=True)

# Features usable at forecast time (forecast days are non-contiguous, so only lag1-based
# and weather/seasonal signals are servable; rolling stats live in the FG for history).
SERVABLE = (
    ["pm25_lag1"]
    + WEATHER
    + ["doy_sin", "doy_cos", "month", "temp_x_wind", "humidity_x_precip", "lag1_x_lowwind"]
)

# ---------------- 2. Training dataset airqtd754fa9 ----------------
query = fg.select(["date"] + SERVABLE + ["pm25"])
fv = fs.get_or_create_feature_view(
    name="airqtd754fa9",
    version=1,
    query=query,
    labels=["pm25"],
    description="Training view for PM2.5 regressor airqmodel754fa9",
)
train_df = None
try:
    td_version, _ = fv.create_training_data(
        description="airqtd754fa9 training dataset",
        write_options={"wait_for_job": True},
    )
    print("training dataset version:", td_version, flush=True)
    X_td, y_td = fv.get_training_data(td_version)
    train_df = pd.concat([X_td.reset_index(drop=True), y_td.reset_index(drop=True)], axis=1)
except Exception as e:  # noqa: BLE001
    print("WARN: reading training dataset back failed, falling back to engineered frame:", e, flush=True)

if train_df is None or len(train_df) < len(fg_df) // 2:
    train_df = fg_df[["date"] + SERVABLE + ["pm25"]].copy()

train_df = train_df.sort_values("date").reset_index(drop=True)
print("training rows:", len(train_df), flush=True)

# ---------------- 3. Train + register model airqmodel754fa9 ----------------
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X_all = train_df[SERVABLE].astype(float).values
y_all = train_df["pm25"].astype(float).values

n_val = 150
X_tr, y_tr = X_all[:-n_val], y_all[:-n_val]
X_va, y_va = X_all[-n_val:], y_all[-n_val:]

candidates = {
    "gbr": GradientBoostingRegressor(
        n_estimators=600, learning_rate=0.03, max_depth=3, subsample=0.9, random_state=42
    ),
    "rf": RandomForestRegressor(n_estimators=500, min_samples_leaf=3, random_state=42, n_jobs=-1),
    "ridge": Ridge(alpha=1.0),
}
results = {}
for name, mdl in candidates.items():
    mdl.fit(X_tr, y_tr)
    rmse = float(np.sqrt(mean_squared_error(y_va, mdl.predict(X_va))))
    results[name] = rmse
    print(f"holdout RMSE [{name}]: {rmse:.4f}", flush=True)

best_name = min(results, key=results.get)
best = candidates[best_name]
va_pred = best.predict(X_va)
metrics = {
    "rmse": results[best_name],
    "mae": float(mean_absolute_error(y_va, va_pred)),
    "r2": float(r2_score(y_va, va_pred)),
}
print("best model:", best_name, "metrics:", metrics, flush=True)

# refit best model on the full history for final predictions
final_model = candidates[best_name].__class__(**candidates[best_name].get_params())
final_model.fit(X_all, y_all)

import joblib

model_dir = "airq_model_dir"
os.makedirs(model_dir, exist_ok=True)
joblib.dump(final_model, os.path.join(model_dir, "model.pkl"))
with open(os.path.join(model_dir, "features.txt"), "w") as f:
    f.write("\n".join(SERVABLE))

mr = project.get_model_registry()
model_meta = mr.python.create_model(
    name="airqmodel754fa9",
    metrics=metrics,
    description=f"PM2.5 daily regressor ({best_name}) trained on airqtd754fa9; holdout = last {n_val} days",
)
model_meta.save(model_dir)
print("model registered: airqmodel754fa9", flush=True)

# ---------------- 4. Predictions feature group airqpred754fa9 ----------------
fc_e = engineer(fc)
preds = final_model.predict(fc_e[SERVABLE].astype(float).values)
pred_df = pd.DataFrame(
    {"date": fc["date"].dt.strftime("%Y-%m-%d"), "pm25_pred": preds.astype(float)}
)
print(pred_df.head(10).to_string(), flush=True)

pred_fg = fs.get_or_create_feature_group(
    name="airqpred754fa9",
    version=1,
    primary_key=["date"],
    description="PM2.5 predictions for forecast days (online-enabled for low-latency lookup)",
    online_enabled=True,
)
pred_fg.insert(pred_df, write_options={"wait_for_job": True})
print("predictions inserted:", len(pred_df), "rows", flush=True)
print("PIPELINE_DONE", flush=True)
