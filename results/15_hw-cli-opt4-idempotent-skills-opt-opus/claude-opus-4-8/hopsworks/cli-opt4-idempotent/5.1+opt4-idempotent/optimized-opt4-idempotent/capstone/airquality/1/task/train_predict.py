"""FTI training + batch inference job — runs ON the Hopsworks platform.

Reads the PM2.5 feature view, trains a regressor, registers it with metrics,
predicts the forecast days, and writes an online+offline predictions FG.
"""
import os
import math
import numpy as np
import pandas as pd
import joblib

import hopsworks
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

FEATS = ["pm25_lag1", "temperature", "humidity", "wind_speed", "pressure", "precipitation"]
LABEL = "pm25"

project = hopsworks.login()
fs = project.get_feature_store()

# ---- read training data (point-in-time-correct TD v1 from the feature view) ----
fv = fs.get_feature_view(name="airqtd7f3c9a", version=1)
try:
    X_train, X_test, y_train, y_test = fv.get_train_test_split(training_dataset_version=1)
    print("Loaded TD v1 via get_train_test_split")
except Exception as e:
    print("get_train_test_split failed (%s); recomputing split" % e)
    X_train, X_test, y_train, y_test = fv.train_test_split(test_size=0.2)

def feat_frame(df):
    df = df.copy()
    for c in FEATS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[FEATS].astype(float)

Xtr = feat_frame(X_train)
Xte = feat_frame(X_test)
ytr = pd.to_numeric(np.ravel(y_train if not hasattr(y_train, "values") else y_train.values.ravel() if hasattr(y_train, "values") else y_train), errors="coerce")
yte = pd.to_numeric(np.ravel(y_test.values.ravel() if hasattr(y_test, "values") else y_test), errors="coerce")

print("train rows:", len(Xtr), "test rows:", len(Xte))
print("baseline (predict pm25_lag1) test RMSE:",
      math.sqrt(mean_squared_error(yte, Xte["pm25_lag1"])))

candidates = {
    "linreg": LinearRegression(),
    "rf": RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1),
    "gbr": GradientBoostingRegressor(random_state=42),
}
scored = {}
for name, mdl in candidates.items():
    mdl.fit(Xtr, ytr)
    rmse = math.sqrt(mean_squared_error(yte, mdl.predict(Xte)))
    scored[name] = rmse
    print(f"model {name}: test RMSE = {rmse:.4f}")

best_name = min(scored, key=scored.get)
print("best model:", best_name, "RMSE:", scored[best_name])

# held-out metrics from the best model
best = candidates[best_name]
pred_te = best.predict(Xte)
rmse = float(math.sqrt(mean_squared_error(yte, pred_te)))
mae = float(mean_absolute_error(yte, pred_te))
r2 = float(r2_score(yte, pred_te))

# refit the best configuration on ALL labelled data for the final model
Xall = pd.concat([Xtr, Xte], axis=0)
yall = np.concatenate([ytr, yte])
final = candidates[best_name].__class__(**candidates[best_name].get_params())
final.fit(Xall, yall)

# ---- register model with metrics ----
mr = project.get_model_registry()
model_dir = "airqmodel_dir"
os.makedirs(model_dir, exist_ok=True)
joblib.dump(final, os.path.join(model_dir, "model.pkl"))
input_example = Xte.head(1).to_dict(orient="records")[0]
metrics = {"rmse": rmse, "mae": mae, "r2": r2}
print("registering metrics:", metrics)

hmodel = mr.python.create_model(
    name="airqmodel7f3c9a",
    metrics=metrics,
    description=f"PM2.5 regressor ({best_name}); held-out RMSE={rmse:.4f}",
    input_example=input_example,
    feature_view=fv,
)
hmodel.save(model_dir)
print("model registered:", hmodel.name, "v", hmodel.version)

# ---- batch inference on forecast days ----
fc = fs.get_feature_group("airqfcin7f3c9a", version=1).read()
fc = fc.sort_values("date").reset_index(drop=True)
Xfc = feat_frame(fc)
preds = final.predict(Xfc).astype(float)
out = pd.DataFrame({"date": fc["date"].astype(str).values, "pm25_pred": preds})
print("forecast predictions:", len(out), "rows; sample:")
print(out.head().to_string())

# ---- write predictions FG (online + offline for low-latency lookup) ----
pred_fg = fs.get_or_create_feature_group(
    name="airqpred7f3c9a",
    version=1,
    primary_key=["date"],
    online_enabled=True,
    description="PM2.5 forecast predictions (online+offline lookup)",
)
pred_fg.insert(out)
print("predictions written to airqpred7f3c9a:", len(out), "rows")
print("DONE")
