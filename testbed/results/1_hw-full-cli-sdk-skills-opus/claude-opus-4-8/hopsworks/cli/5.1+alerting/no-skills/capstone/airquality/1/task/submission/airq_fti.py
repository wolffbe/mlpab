"""Air-quality FTI: train+register PM2.5 regressor, predict forecast days, write online FG."""
import os
import numpy as np
import pandas as pd
import joblib
import hopsworks
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

FEATURES = ["pm25_lag1", "temperature", "humidity", "wind_speed", "pressure", "precipitation"]
LABEL = "pm25"

project = hopsworks.login()
fs = project.get_feature_store()

# ---- Read the materialized training dataset from the feature view ----
fv = fs.get_feature_view("airqtd963ee7", version=1)
X_train, X_test, y_train, y_test = fv.get_train_test_split(training_dataset_version=1)


def as_X(df):
    return df[FEATURES].astype(float)


def as_y(y):
    if hasattr(y, "columns"):
        return y[LABEL].astype(float)
    return pd.Series(y).astype(float)


Xtr, Xte = as_X(X_train), as_X(X_test)
ytr, yte = as_y(y_train), as_y(y_test)

params = dict(n_estimators=500, max_depth=3, learning_rate=0.05,
              subsample=0.9, random_state=42)

# ---- Held-out evaluation model ----
eval_model = GradientBoostingRegressor(**params)
eval_model.fit(Xtr, ytr)
pred_te = eval_model.predict(Xte)
rmse = float(np.sqrt(mean_squared_error(yte, pred_te)))
mae = float(mean_absolute_error(yte, pred_te))
print(f"HELD-OUT RMSE={rmse:.4f}  MAE={mae:.4f}")

# ---- Final model trained on all available history ----
Xfull = pd.concat([Xtr, Xte], ignore_index=True)
yfull = pd.concat([ytr, yte], ignore_index=True)
final_model = GradientBoostingRegressor(**params)
final_model.fit(Xfull, yfull)

# ---- Register the model with metrics ----
mr = project.get_model_registry()
mdir = "airqmodel963ee7"
os.makedirs(mdir, exist_ok=True)
joblib.dump(final_model, os.path.join(mdir, "model.pkl"))
input_example = Xfull.head(1).to_dict(orient="records")[0]
model = mr.python.create_model(
    name="airqmodel963ee7",
    metrics={"rmse": rmse, "mae": mae},
    description="PM2.5 daily regressor (GradientBoosting) on weather + pm25_lag1",
    feature_view=fv,
    training_dataset_version=1,
    input_example=input_example,
)
model.save(mdir)
print("Registered model airqmodel963ee7 with metrics", {"rmse": rmse, "mae": mae})

# ---- Predict forecast days ----
ds = project.get_dataset_api()
local_fc = ds.download("Resources/airq/forecast_days.csv", overwrite=True)
fc = pd.read_csv(local_fc)
preds = final_model.predict(fc[FEATURES].astype(float))
out = pd.DataFrame({
    "date": fc["date"].astype(str),
    "pm25_pred": np.asarray(preds, dtype=float),
})
print("Predicted rows:", len(out))

# ---- Write predictions to an online-enabled feature group ----
pred_fg = fs.get_or_create_feature_group(
    name="airqpred963ee7",
    version=1,
    primary_key=["date"],
    online_enabled=True,
    description="PM2.5 forecast predictions (online + offline)",
)
pred_fg.insert(out)
print("Inserted predictions into airqpred963ee7")
