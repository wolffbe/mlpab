"""Full FTI pipeline for PM2.5 forecasting — runs on the Hopsworks platform as a job.

Feature -> Training -> Inference, all platform-side:
  1. Engineer features into feature group `airqc850d2`.
  2. Assemble a training dataset via feature view `airqtdc850d2`.
  3. Train + register regressor `airqmodelc850d2` (with metrics).
  4. Predict forecast_days into feature table `airqpredc850d2` (online + offline).
"""
import os
import numpy as np
import pandas as pd
import hopsworks

FG_NAME = "airqc850d2"
FV_NAME = "airqtdc850d2"
MODEL_NAME = "airqmodelc850d2"
PRED_FG = "airqpredc850d2"

# Features the model uses — all available for forecast rows too.
MODEL_FEATURES = ["pm25_lag1", "temperature", "humidity",
                  "wind_speed", "pressure", "precipitation"]

print("=== Logging in ===", flush=True)
project = hopsworks.login()
fs = project.get_feature_store()
dsapi = project.get_dataset_api()

# --- Fetch the uploaded CSVs into the job's working dir ---
def fetch(remote, local):
    if os.path.exists(local):
        os.remove(local)
    dsapi.download(remote, local_path=local, overwrite=True)
    return pd.read_csv(local)

hist = fetch("Resources/airq/airquality_history.csv", "hist.csv")
fcst = fetch("Resources/airq/forecast_days.csv", "fcst.csv")
print(f"history rows={len(hist)} forecast rows={len(fcst)}", flush=True)

# =====================================================================
# 1. FEATURE ENGINEERING -> feature group airqc850d2
# =====================================================================
def engineer(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    # weather-derived signals (available per-row at inference)
    df["temp_humidity"] = df["temperature"] * df["humidity"] / 100.0
    df["wind_precip"] = df["wind_speed"] * (df["precipitation"] + 1.0)
    return df

hist_fe = engineer(hist)
# rolling air-quality signals on the contiguous history (engineered into the FG)
hist_fe["pm25_roll3"] = hist_fe["pm25"].shift(1).rolling(3, min_periods=1).mean()
hist_fe["pm25_roll7"] = hist_fe["pm25"].shift(1).rolling(7, min_periods=1).mean()
hist_fe["pm25_roll3"] = hist_fe["pm25_roll3"].fillna(hist_fe["pm25_lag1"])
hist_fe["pm25_roll7"] = hist_fe["pm25_roll7"].fillna(hist_fe["pm25_lag1"])

print("=== Creating feature group airqc850d2 ===", flush=True)
fg = fs.get_or_create_feature_group(
    name=FG_NAME,
    version=1,
    description="Engineered daily air-quality + weather features (target pm25).",
    primary_key=["date"],
    event_time="date",
    online_enabled=True,
)
fg.insert(hist_fe, write_options={"wait_for_job": True})
print("Inserted history into FG.", flush=True)

# =====================================================================
# 2. TRAINING DATASET -> feature view airqtdc850d2
# =====================================================================
print("=== Creating feature view + training dataset airqtdc850d2 ===", flush=True)
try:
    existing = fs.get_feature_view(name=FV_NAME, version=1)
    existing.delete()
except Exception:
    pass

query = fg.select_all()
fv = fs.create_feature_view(
    name=FV_NAME,
    version=1,
    description="Training dataset for PM2.5 regressor (label pm25).",
    query=query,
    labels=["pm25"],
)

td_version, _ = fv.create_train_test_split(
    test_size=0.2,
    description="airqtdc850d2 train/test materialization",
    write_options={"wait_for_job": True},
)
print(f"Materialized training dataset version={td_version}", flush=True)

# Read the materialized training data back from the platform.
try:
    X_train, X_test, y_train, y_test = fv.get_train_test_split(td_version)
    print(f"Read TD: train={len(X_train)} test={len(X_test)}", flush=True)
except Exception as e:
    print(f"TD read fallback ({e}); using in-job split.", flush=True)
    from sklearn.model_selection import train_test_split
    Xall = hist_fe.drop(columns=["pm25"])
    yall = hist_fe["pm25"]
    X_train, X_test, y_train, y_test = train_test_split(
        Xall, yall, test_size=0.2, shuffle=False)

# keep only model feature columns
def feats(df):
    return df[MODEL_FEATURES].astype(float)

Xtr, Xte = feats(X_train), feats(X_test)
ytr = np.asarray(y_train).ravel().astype(float)
yte = np.asarray(y_test).ravel().astype(float)

# =====================================================================
# 3. TRAIN + REGISTER model airqmodelc850d2
# =====================================================================
from sklearn.ensemble import (GradientBoostingRegressor,
                              HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

candidates = {
    "gbr": GradientBoostingRegressor(n_estimators=400, max_depth=3,
                                     learning_rate=0.03, subsample=0.9,
                                     random_state=42),
    "hgb": HistGradientBoostingRegressor(max_iter=500, learning_rate=0.03,
                                         max_depth=4, random_state=42),
    "rf": RandomForestRegressor(n_estimators=500, max_depth=8,
                                random_state=42, n_jobs=-1),
}

best_name, best_model, best_rmse = None, None, float("inf")
for name, mdl in candidates.items():
    mdl.fit(Xtr, ytr)
    pred = mdl.predict(Xte)
    rmse = float(np.sqrt(mean_squared_error(yte, pred)))
    print(f"  {name}: holdout RMSE={rmse:.4f}", flush=True)
    if rmse < best_rmse:
        best_name, best_model, best_rmse = name, mdl, rmse

print(f"Best model: {best_name} (holdout RMSE={best_rmse:.4f})", flush=True)
pred_te = best_model.predict(Xte)
metrics = {
    "rmse": round(float(np.sqrt(mean_squared_error(yte, pred_te))), 4),
    "mae": round(float(mean_absolute_error(yte, pred_te)), 4),
    "r2": round(float(r2_score(yte, pred_te)), 4),
}
print(f"Held-out metrics: {metrics}", flush=True)

# Refit best estimator on ALL history for the strongest forecast model.
Xfull = feats(hist_fe.drop(columns=["pm25"]) if "pm25" in hist_fe else hist_fe)
final_model = best_model.__class__(**best_model.get_params())
final_model.fit(
    hist_fe[MODEL_FEATURES].astype(float),
    hist_fe["pm25"].astype(float).values,
)

# Save + register the model
import joblib
mr = project.get_model_registry()
model_dir = "airq_model"
os.makedirs(model_dir, exist_ok=True)
joblib.dump(final_model, os.path.join(model_dir, "model.pkl"))

input_example = hist_fe[MODEL_FEATURES].iloc[:1].astype(float)
try:
    from hsml.schema import Schema
    from hsml.model_schema import ModelSchema
    model_schema = ModelSchema(
        input_schema=Schema(input_example),
        output_schema=Schema(hist_fe[["pm25"]]),
    )
except Exception as e:
    print(f"schema build skipped: {e}", flush=True)
    model_schema = None

hops_model = mr.python.create_model(
    name=MODEL_NAME,
    metrics=metrics,
    description="PM2.5 daily regressor (best of GBR/HGB/RF).",
    input_example=input_example,
    model_schema=model_schema,
    feature_view=fv,
)
hops_model.save(model_dir)
print(f"Registered model {MODEL_NAME} v{hops_model.version} metrics={metrics}", flush=True)

# =====================================================================
# 4. INFERENCE -> predictions feature table airqpredc850d2
# =====================================================================
print("=== Predicting forecast_days ===", flush=True)
fcst_fe = engineer(fcst)
Xf = fcst_fe[MODEL_FEATURES].astype(float)
preds = final_model.predict(Xf)

pred_df = pd.DataFrame({
    "date": pd.to_datetime(fcst["date"]).dt.strftime("%Y-%m-%d"),
    "pm25_pred": preds.astype(float),
})
print(pred_df.head(10).to_string(), flush=True)

pred_fg = fs.get_or_create_feature_group(
    name=PRED_FG,
    version=1,
    description="PM2.5 forecast predictions (pm25_pred) keyed by date.",
    primary_key=["date"],
    online_enabled=True,
)
pred_fg.insert(pred_df, write_options={"wait_for_job": True})
print(f"Wrote {len(pred_df)} predictions into {PRED_FG} (online+offline).", flush=True)
print("=== PIPELINE COMPLETE ===", flush=True)
