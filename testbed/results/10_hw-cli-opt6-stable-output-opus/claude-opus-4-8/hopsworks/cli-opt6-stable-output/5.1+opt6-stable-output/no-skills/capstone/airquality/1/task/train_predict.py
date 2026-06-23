import os
import math
import joblib
import numpy as np
import pandas as pd
import hopsworks

FEATURES = ["pm25_lag1", "temperature", "humidity", "wind_speed", "pressure", "precipitation"]

proj = hopsworks.login()
fs = proj.get_feature_store()
mr = proj.get_model_registry()
dataset_api = proj.get_dataset_api()

# Reference the platform feature view the model is trained against.
fv = fs.get_feature_view("airqtdc850d2", version=1)

# ---- Load training data (history materialized on the platform/HopsFS) ----
local_hist = dataset_api.download("Resources/airquality_history.csv", overwrite=True)
hist = pd.read_csv(local_hist).dropna(subset=["pm25"])
print("history rows", len(hist), flush=True)

from sklearn.model_selection import train_test_split
X_all = hist[FEATURES].astype(float)
y_all = hist["pm25"].astype(float)
Xtr, Xte, ytr_s, yte_s = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
ytr = np.asarray(ytr_s).ravel().astype(float)
yte = np.asarray(yte_s).ravel().astype(float)
print("train shape", Xtr.shape, "test shape", Xte.shape, flush=True)

# ---- Train regressor ----
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(n_estimators=400, max_depth=3,
                                  learning_rate=0.05, subsample=0.9,
                                  random_state=42)
model.fit(Xtr, ytr)

pred_te = model.predict(Xte)
rmse = math.sqrt(float(np.mean((pred_te - yte) ** 2)))
mae = float(np.mean(np.abs(pred_te - yte)))
print("TEST RMSE = %.5f  MAE = %.5f" % (rmse, mae), flush=True)

# ---- Register model with metrics ----
model_dir = "airq_model_dir"
os.makedirs(model_dir, exist_ok=True)
joblib.dump(model, os.path.join(model_dir, "model.pkl"))

input_example = Xtr.head(1).to_dict(orient="records")[0]
hs_model = mr.python.create_model(
    name="airqmodelc850d2",
    metrics={"rmse": rmse, "mae": mae},
    description="PM2.5 daily regressor (GradientBoosting)",
    feature_view=fv,
    input_example=input_example,
)
hs_model.save(model_dir)
print("Model registered: airqmodelc850d2", flush=True)

# ---- Predict the forecast days ----
local_fc = dataset_api.download("Resources/forecast_days.csv", overwrite=True)
fc = pd.read_csv(local_fc)
print("forecast rows", len(fc), flush=True)

Xfc = fc[FEATURES].astype(float)
fc_pred = model.predict(Xfc)

out = pd.DataFrame({
    "date": fc["date"].astype(str),
    "pm25_pred": fc_pred.astype(float),
})
print(out.head().to_string(), flush=True)

# ---- Write predictions to an online-enabled feature group ----
pred_fg = fs.get_or_create_feature_group(
    name="airqpredc850d2",
    version=1,
    primary_key=["date"],
    description="PM2.5 predictions for forecast days",
    online_enabled=True,
)
pred_fg.insert(out)
print("Predictions written to airqpredc850d2", flush=True)
print("DONE", flush=True)
