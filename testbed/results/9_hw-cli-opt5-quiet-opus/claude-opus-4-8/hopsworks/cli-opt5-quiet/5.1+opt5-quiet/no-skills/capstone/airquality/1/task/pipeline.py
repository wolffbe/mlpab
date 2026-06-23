"""Full FTI pipeline for PM2.5 forecasting — runs as a Hopsworks job (on-platform)."""
import os
import numpy as np
import pandas as pd
import hopsworks
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

print("=== connecting ===", flush=True)
proj = hopsworks.login()
fs = proj.get_feature_store()
dsapi = proj.get_dataset_api()

# ---- load raw inputs from HopsFS ----
for remote, local in [
    ("Resources/airq/airquality_history.csv", "history.csv"),
    ("Resources/airq/forecast_days.csv", "forecast.csv"),
]:
    if os.path.exists(local):
        os.remove(local)
    dsapi.download(remote, local)

hist = pd.read_csv("history.csv")
fc = pd.read_csv("forecast.csv")
print("history", hist.shape, "forecast", fc.shape, flush=True)

PRED = ["pm25_lag1", "temperature", "humidity", "wind_speed",
        "pressure", "precipitation", "doy_sin", "doy_cos"]


def engineer(df):
    df = df.copy()
    df["date"] = df["date"].astype(str)
    dt = pd.to_datetime(df["date"])
    df["event_time"] = dt
    doy = dt.dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    for c in ["pm25_lag1", "temperature", "humidity", "wind_speed",
              "pressure", "precipitation"]:
        df[c] = df[c].astype(float)
    return df


hist_f = engineer(hist)
hist_f["pm25"] = hist_f["pm25"].astype(float)
fc_f = engineer(fc)

# ================= FEATURE GROUP =================
print("=== feature group airqeda516 ===", flush=True)
fg_cols = ["date", "event_time"] + PRED + ["pm25"]
fg = fs.get_or_create_feature_group(
    name="airqeda516", version=1,
    description="Engineered air-quality features (weather + lag + seasonality) with pm25 target",
    primary_key=["date"], event_time="event_time", online_enabled=False,
)
fg.insert(hist_f[fg_cols], write_options={"wait_for_job": True})
print("inserted FG rows:", len(hist_f), flush=True)

# ================= FEATURE VIEW + TRAINING DATASET =================
print("=== feature view / training dataset airqtdeda516 ===", flush=True)
try:
    fs.get_feature_view("airqtdeda516", version=1).delete()
except Exception:
    pass
query = fg.select(PRED + ["pm25"])
fv = fs.create_feature_view(
    name="airqtdeda516", version=1, query=query, labels=["pm25"],
    description="Training view for PM2.5 regressor",
)
td_version, _ = fv.create_training_data(
    description="airqtdeda516 training dataset", write_options={"wait_for_job": True})
print("training dataset version:", td_version, flush=True)

# ================= TRAIN =================
print("=== train ===", flush=True)
d = hist_f.sort_values("event_time").reset_index(drop=True)
X = d[PRED].values
y = d["pm25"].values
n = len(d)
cut = int(n * 0.8)
Xtr, Xte, ytr, yte = X[:cut], X[cut:], y[:cut], y[cut:]

model = RandomForestRegressor(n_estimators=400, min_samples_leaf=2,
                              random_state=42, n_jobs=-1)
model.fit(Xtr, ytr)
rmse_holdout = float(np.sqrt(mean_squared_error(yte, model.predict(Xte))))
print("HELD-OUT RMSE:", rmse_holdout, flush=True)

# refit on all data for the deployed model / forecast
final_model = RandomForestRegressor(n_estimators=400, min_samples_leaf=2,
                                    random_state=42, n_jobs=-1)
final_model.fit(X, y)

# ================= REGISTER MODEL =================
print("=== register model airqmodeleda516 ===", flush=True)
import joblib
os.makedirs("model_dir", exist_ok=True)
joblib.dump(final_model, "model_dir/model.pkl")
input_example = d[PRED].iloc[:1].to_dict(orient="records")[0]

mr = proj.get_model_registry()
m = mr.python.create_model(
    name="airqmodeleda516",
    metrics={"rmse": rmse_holdout},
    description="RandomForest PM2.5 regressor",
    input_example=input_example,
    feature_view=fv,
    training_dataset_version=td_version,
)
m.save("model_dir")
print("registered model version:", m.version, flush=True)

# ================= PREDICT =================
print("=== predict -> airqprededa516 ===", flush=True)
preds = final_model.predict(fc_f[PRED].values)
out = pd.DataFrame({"date": fc_f["date"].astype(str).values,
                    "pm25_pred": preds.astype(float)})
print(out.head().to_string(), flush=True)

pred_fg = fs.get_or_create_feature_group(
    name="airqprededa516", version=1,
    description="PM2.5 predictions for forecast days",
    primary_key=["date"], online_enabled=True,
)
pred_fg.insert(out, write_options={"wait_for_job": True})
print("inserted predictions:", len(out), flush=True)
print("DONE rmse=", rmse_holdout, flush=True)
