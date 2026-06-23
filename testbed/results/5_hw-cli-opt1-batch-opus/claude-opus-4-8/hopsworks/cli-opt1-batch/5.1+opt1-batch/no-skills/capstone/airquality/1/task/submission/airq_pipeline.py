"""Full FTI pipeline for PM2.5 forecasting, run as a Hopsworks PYTHON job.

Runs entirely on the platform:
  1. Feature engineering -> feature group airq854125
  2. Feature view + training dataset airqtd854125
  3. Train + register sklearn regressor airqmodel854125 (with metrics)
  4. Predict forecast_days -> feature group airqpred854125 (online + offline)
"""
import os
import numpy as np
import pandas as pd

import hopsworks
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

FEATURES = ["pm25_lag1", "temperature", "humidity",
            "wind_speed", "pressure", "precipitation"]
LABEL = "pm25"

print(">>> logging in")
proj = hopsworks.login()
fs = proj.get_feature_store()
ds = proj.get_dataset_api()

# ---------------------------------------------------------------- load inputs
ds.download("Resources/airq854125/airquality_history.csv",
            local_path="hist.csv", overwrite=True)
ds.download("Resources/airq854125/forecast_days.csv",
            local_path="fc.csv", overwrite=True)
hist = pd.read_csv("hist.csv")
fc = pd.read_csv("fc.csv")
print(">>> history rows:", len(hist), "forecast rows:", len(fc))

# ensure clean numeric dtypes
for c in FEATURES + [LABEL]:
    if c in hist.columns:
        hist[c] = pd.to_numeric(hist[c], errors="coerce")
for c in FEATURES:
    fc[c] = pd.to_numeric(fc[c], errors="coerce")
hist["date"] = hist["date"].astype(str)
fc["date"] = fc["date"].astype(str)

# ----------------------------------------------------- 1. FEATURE GROUP (FTI: F)
print(">>> creating feature group airq854125")
fg = fs.get_or_create_feature_group(
    name="airq854125",
    version=1,
    primary_key=["date"],
    description="Daily air-quality history: weather + pm25 lag-1 signal and measured pm25",
    online_enabled=True,
)
fg.insert(hist, write_options={"wait_for_job": True})
print(">>> feature group inserted")

# ------------------------------------------- 2. FEATURE VIEW + TRAINING DATASET
print(">>> creating feature view + training dataset airqtd854125")
try:
    fv = fs.get_feature_view(name="airqtd854125", version=1)
    print(">>> feature view already existed")
except Exception:
    query = fg.select_all()
    fv = fs.create_feature_view(
        name="airqtd854125",
        version=1,
        query=query,
        labels=[LABEL],
        description="Feature view for PM2.5 regression (label=pm25)",
    )
    print(">>> feature view created")

td_version = 1
try:
    res = fv.create_train_test_split(
        test_size=0.2,
        description="airq train/test split",
        write_options={"wait_for_job": True},
    )
    # API returns (version, job) in recent hsfs
    if isinstance(res, (tuple, list)):
        td_version = res[0]
    print(">>> training dataset materialized, version:", td_version)
except Exception as e:
    print(">>> WARN training dataset creation:", repr(e))

# --------------------------------------------------------- 3. TRAIN + REGISTER
# In-memory split for a reliable held-out metric (independent of offline
# materialization timing). The persisted TD above is the platform deliverable.
X = hist[FEATURES]
y = hist[LABEL]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

model = HistGradientBoostingRegressor(
    max_iter=600,
    learning_rate=0.05,
    max_depth=4,
    min_samples_leaf=15,
    l2_regularization=1.0,
    random_state=42,
)
model.fit(Xtr, ytr)
pred_te = model.predict(Xte)
rmse = float(np.sqrt(mean_squared_error(yte, pred_te)))
mae = float(mean_absolute_error(yte, pred_te))
r2 = float(r2_score(yte, pred_te))
print(f">>> HELD-OUT RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")

# refit on all available history for the strongest forecast model
model.fit(X, y)

import joblib
mr = proj.get_model_registry()
mdir = "airqmodel854125"
os.makedirs(mdir, exist_ok=True)
joblib.dump(model, os.path.join(mdir, "model.pkl"))

input_example = Xtr.head(1).to_dict(orient="records")[0]
metrics = {"rmse": rmse, "mae": mae, "r2": r2}

mm = None
for kwargs in (
    dict(name="airqmodel854125", metrics=metrics, description="PM2.5 daily regressor",
         input_example=input_example, feature_view=fv),
    dict(name="airqmodel854125", metrics=metrics, description="PM2.5 daily regressor",
         input_example=input_example),
    dict(name="airqmodel854125", metrics=metrics, description="PM2.5 daily regressor"),
):
    try:
        mm = mr.sklearn.create_model(**kwargs)
        break
    except Exception as e:
        print(">>> WARN create_model variant failed:", repr(e))
if mm is None:
    raise RuntimeError("could not create model in registry")
mm.save(mdir)
print(">>> model registered: airqmodel854125 version", getattr(mm, "version", "?"))

# ------------------------------------------------- 4. PREDICT -> PREDICTION FG
print(">>> predicting forecast days")
fc_out = fc.copy()
fc_out["pm25_pred"] = model.predict(fc_out[FEATURES]).astype(float)
out = fc_out[["date", "pm25_pred"]].copy()
out["date"] = out["date"].astype(str)
print(out.head(10).to_string())

predfg = fs.get_or_create_feature_group(
    name="airqpred854125",
    version=1,
    primary_key=["date"],
    description="PM2.5 predictions for forecast days (online + offline)",
    online_enabled=True,
)
predfg.insert(out, write_options={"wait_for_job": True})
print(">>> predictions inserted:", len(out))
print(">>> DONE")
