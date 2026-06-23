"""Full FTI pipeline for PM2.5 forecasting, runs as a Hopsworks job on the platform."""
import os
import numpy as np
import pandas as pd
import joblib
import hopsworks
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print(">>> logging in", flush=True)
project = hopsworks.login()
fs = project.get_feature_store()
dsapi = project.get_dataset_api()

# ---- 1. fetch raw data uploaded to HopsFS ----
for f in ["airquality_history.csv", "forecast_days.csv"]:
    if os.path.exists(f):
        os.remove(f)
    dsapi.download(f"Resources/airq/{f}", local_path=".", overwrite=True)

hist = pd.read_csv("airquality_history.csv")
fcst = pd.read_csv("forecast_days.csv")
print(">>> history", hist.shape, "forecast", fcst.shape, flush=True)

RAW = ["pm25_lag1", "temperature", "humidity", "wind_speed", "pressure", "precipitation"]
FEATS = RAW + ["month", "doy_sin", "doy_cos", "temp_hum", "wind_press"]


def engineer(df):
    df = df.copy()
    dt = pd.to_datetime(df["date"])
    df["month"] = dt.dt.month.astype(float)
    doy = dt.dt.dayofyear.astype(float)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    for c in RAW:
        df[c] = df[c].astype(float)
    df["temp_hum"] = df["temperature"] * df["humidity"] / 100.0
    df["wind_press"] = df["wind_speed"] * df["pressure"] / 1000.0
    df["date"] = df["date"].astype(str)
    return df


hist_f = engineer(hist)
fcst_f = engineer(fcst)

# ---- 2. feature group: engineered features ----
print(">>> creating feature group airq1b5b85", flush=True)
airq_fg = fs.get_or_create_feature_group(
    name="airq1b5b85",
    version=1,
    description="Engineered air-quality features (weather + lag signals) for PM2.5 forecasting",
    primary_key=["date"],
    online_enabled=True,
)
airq_fg.insert(hist_f[["date"] + FEATS + ["pm25"]], write_options={"wait_for_job": True})
print(">>> feature group populated", flush=True)

# ---- 3. feature view + training dataset ----
print(">>> creating feature view / training dataset airqtd1b5b85", flush=True)
try:
    existing = fs.get_feature_view(name="airqtd1b5b85", version=1)
    existing.delete()
except Exception as e:
    print("no existing fv:", e, flush=True)

fv = fs.create_feature_view(
    name="airqtd1b5b85",
    version=1,
    description="Training feature view for PM2.5 regressor",
    query=airq_fg.select(FEATS + ["pm25"]),
    labels=["pm25"],
)
# materialize a persisted training dataset version
fv.create_training_data(
    description="PM2.5 training dataset",
    data_format="csv",
    write_options={"wait_for_job": True},
)
print(">>> training dataset materialized", flush=True)

# ---- 4. train regressor on engineered history ----
X = hist_f[FEATS]
y = hist_f["pm25"].astype(float)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(
    n_estimators=400, learning_rate=0.05, max_depth=3, subsample=0.9, random_state=42
)
model.fit(X_tr, y_tr)
pred_te = model.predict(X_te)
rmse = float(np.sqrt(mean_squared_error(y_te, pred_te)))
mae = float(mean_absolute_error(y_te, pred_te))
r2 = float(r2_score(y_te, pred_te))
print(f">>> HELDOUT RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f}", flush=True)

# ---- 5. register model with metrics ----
print(">>> registering model airqmodel1b5b85", flush=True)
mr = project.get_model_registry()
os.makedirs("airq_model", exist_ok=True)
joblib.dump(model, "airq_model/model.pkl")
with open("airq_model/features.txt", "w") as fh:
    fh.write(",".join(FEATS))

input_example = X_tr.head(2)
metrics = {"rmse": rmse, "mae": mae, "r2": r2}
hmodel = mr.python.create_model(
    name="airqmodel1b5b85",
    metrics=metrics,
    description="GradientBoosting PM2.5 regressor",
    input_example=input_example,
)
hmodel.save("airq_model")
print(">>> model registered with metrics", metrics, flush=True)

# ---- 6. predict forecast days into prediction feature table ----
print(">>> predicting forecast days", flush=True)
fcst_pred = model.predict(fcst_f[FEATS])
out = pd.DataFrame({"date": fcst["date"].astype(str), "pm25_pred": fcst_pred.astype(float)})
print(out.head(), flush=True)

pred_fg = fs.get_or_create_feature_group(
    name="airqpred1b5b85",
    version=1,
    description="PM2.5 predictions for forecast days",
    primary_key=["date"],
    online_enabled=True,
)
pred_fg.insert(out, write_options={"wait_for_job": True})
print(">>> predictions written to airqpred1b5b85 (online+offline)", flush=True)
print(">>> PIPELINE COMPLETE", flush=True)
