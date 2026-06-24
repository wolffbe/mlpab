"""FTI training + inference job — runs ON the Hopsworks platform.

Reads the training dataset from feature view airqtde09430, trains a PM2.5
regressor, registers it as airqmodele09430 with metrics, predicts every
forecast row, and writes predictions to online+offline FG airqprede09430.
"""
import os
import math
import numpy as np
import pandas as pd
import hopsworks

FEATURES = ["pm25_lag1", "temperature", "humidity", "wind_speed", "pressure", "precipitation"]
LABEL = "pm25"

proj = hopsworks.login()
fs = proj.get_feature_store()

fv = fs.get_feature_view(name="airqtde09430", version=1)

# ---- read materialized training dataset (v1), fall back to fresh split ----
try:
    X_train, X_test, y_train, y_test = fv.get_train_test_split(training_dataset_version=1)
    print("Loaded train/test split from TD v1")
except Exception as e:
    print("get_train_test_split failed (%s); computing fresh split" % e)
    X_train, X_test, y_train, y_test = fv.train_test_split(test_size=0.2)


def prep(df):
    df = df.copy()
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[FEATURES].astype(float)


Xtr, Xte = prep(X_train), prep(X_test)
ytr = pd.to_numeric(np.ravel(y_train), errors="coerce").astype(float)
yte = pd.to_numeric(np.ravel(y_test), errors="coerce").astype(float)
print("train rows=%d  test rows=%d" % (len(Xtr), len(Xte)))

# ---- train a strong gradient-boosted regressor ----
model = None
try:
    from xgboost import XGBRegressor
    model = XGBRegressor(n_estimators=600, max_depth=4, learning_rate=0.04,
                         subsample=0.9, colsample_bytree=0.9, min_child_weight=2,
                         reg_lambda=1.0, random_state=42, n_jobs=4)
    model.fit(Xtr, ytr)
    print("Trained XGBRegressor")
except Exception as e:
    print("xgboost unavailable (%s); using sklearn HistGradientBoosting" % e)
    from sklearn.ensemble import HistGradientBoostingRegressor
    model = HistGradientBoostingRegressor(max_iter=600, max_depth=4, learning_rate=0.05,
                                          random_state=42)
    model.fit(Xtr, ytr)

pred_te = model.predict(Xte)
rmse = float(math.sqrt(np.mean((pred_te - yte) ** 2)))
mae = float(np.mean(np.abs(pred_te - yte)))
denom = np.sum((yte - yte.mean()) ** 2)
r2 = float(1.0 - np.sum((pred_te - yte) ** 2) / denom) if denom > 0 else 0.0
print("HELD-OUT TEST  rmse=%.4f  mae=%.4f  r2=%.4f" % (rmse, mae, r2))

# ---- register model with metrics ----
import joblib
mr = proj.get_model_registry()
model_dir = "airq_model_dir"
os.makedirs(model_dir, exist_ok=True)
joblib.dump(model, os.path.join(model_dir, "model.pkl"))

input_example = Xtr.head(1).to_dict(orient="records")[0]
try:
    from hsml.schema import Schema
    from hsml.model_schema import ModelSchema
    mschema = ModelSchema(input_schema=Schema(Xtr), output_schema=Schema(ytr.reshape(-1, 1)))
except Exception as e:
    print("model schema build skipped (%s)" % e)
    mschema = None

metrics = {"rmse": rmse, "mae": mae, "r2": r2}
kwargs = dict(name="airqmodele09430", metrics=metrics,
              description="PM2.5 daily regressor (weather + pm25_lag1)",
              input_example=input_example)
if mschema is not None:
    kwargs["model_schema"] = mschema
try:
    kwargs["feature_view"] = fv
    m = mr.python.create_model(**kwargs)
except Exception as e:
    print("create_model with feature_view failed (%s); retrying without" % e)
    kwargs.pop("feature_view", None)
    m = mr.python.create_model(**kwargs)
m.save(model_dir)
print("Registered model airqmodele09430 version", m.version, "metrics", metrics)

# ---- predict forecast rows ----
dsapi = proj.get_dataset_api()
local_csv = dsapi.download("Resources/airq/forecast_days.csv", overwrite=True)
fc = pd.read_csv(local_csv)
print("forecast rows=%d  cols=%s" % (len(fc), list(fc.columns)))
Xf = prep(fc)
fc_pred = model.predict(Xf)
out = pd.DataFrame({"date": fc["date"].astype(str), "pm25_pred": np.asarray(fc_pred, dtype=float)})
print(out.head().to_string())

# ---- write predictions FG (online + offline) ----
pred_fg = fs.get_or_create_feature_group(
    name="airqprede09430", version=1,
    primary_key=["date"],
    online_enabled=True,
    description="PM2.5 predictions for forecast days (online+offline)",
)
pred_fg.insert(out)
print("Inserted %d predictions into airqprede09430" % len(out))
print("DONE")
