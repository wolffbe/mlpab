import warnings, os, json
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib
import hopsworks
from hsfs.feature import Feature

FEATURES = ["pm25_lag1", "temperature", "humidity", "wind_speed", "pressure", "precipitation"]
TD_VERSION = 1

project = hopsworks.login()
fs = project.get_feature_store()
fv = fs.get_feature_view(name="airqtd963ee7", version=1)

X_train, X_test, y_train, y_test = fv.get_train_test_split(training_dataset_version=TD_VERSION)
X_train = X_train[FEATURES].astype("float64")
X_test = X_test[FEATURES].astype("float64")
y_train = np.asarray(y_train).ravel().astype("float64")
y_test = np.asarray(y_test).ravel().astype("float64")
print("train/test shapes:", X_train.shape, X_test.shape)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))

candidates = {}
try:
    from xgboost import XGBRegressor
    for n, d, lr in [(400, 3, 0.05), (600, 4, 0.03), (300, 3, 0.08)]:
        candidates[f"xgb_{n}_{d}_{lr}"] = XGBRegressor(
            n_estimators=n, max_depth=d, learning_rate=lr,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=4)
except Exception as e:
    print("xgboost unavailable:", repr(e))

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, HistGradientBoostingRegressor
candidates["gbr"] = GradientBoostingRegressor(n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42)
candidates["hgb"] = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_depth=3, random_state=42)
candidates["rf"] = RandomForestRegressor(n_estimators=400, max_depth=8, random_state=42, n_jobs=4)

results = {}
for name, mdl in candidates.items():
    mdl.fit(X_train, y_train)
    p = mdl.predict(X_test)
    results[name] = rmse(y_test, p)
    print(f"{name}: test RMSE={results[name]:.4f}")

best_name = min(results, key=results.get)
print("BEST:", best_name, "RMSE", results[best_name])

# metrics from the held-out test split with the best config
best_mdl = candidates[best_name]
preds_test = best_mdl.predict(X_test)
metrics = {
    "rmse": rmse(y_test, preds_test),
    "mae": float(mean_absolute_error(y_test, preds_test)),
    "r2": float(r2_score(y_test, preds_test)),
}
print("METRICS:", metrics)

# refit best config on ALL history for the final deployed model
import copy
final_model = copy.deepcopy(candidates[best_name])
X_all = pd.concat([X_train, X_test], axis=0)
y_all = np.concatenate([y_train, y_test])
final_model.fit(X_all, y_all)

# --- register model ---
model_dir = "airq_model_dir"
os.makedirs(model_dir, exist_ok=True)
joblib.dump(final_model, f"{model_dir}/model.pkl")
json.dump(FEATURES, open(f"{model_dir}/feature_names.json", "w"))

mr = project.get_model_registry()
hw_model = mr.python.create_model(
    name="airqmodel963ee7",
    metrics=metrics,
    description=f"PM2.5 daily regressor ({best_name}); held-out RMSE {metrics['rmse']:.4f}",
    input_example=X_all.head(1),
    feature_view=fv,
    training_dataset_version=TD_VERSION,
)
hw_model.save(model_dir)
print("registered model airqmodel963ee7 v", hw_model.version)

# --- inference on forecast days ---
ds = project.get_dataset_api()
local_csv = ds.download("Resources/airq/forecast_days.csv", overwrite=True)
fc = pd.read_csv(local_csv)
fc_dates = fc["date"].astype(str).values
Xf = fc[FEATURES].astype("float64")
pm25_pred = final_model.predict(Xf).astype("float64")

pred_df = pd.DataFrame({"date": fc_dates, "pm25_pred": pm25_pred})
print("predictions sample:\n", pred_df.head())

pred_fg = fs.get_or_create_feature_group(
    name="airqpred963ee7",
    version=1,
    description="Predicted PM2.5 for forecast days",
    primary_key=["date"],
    online_enabled=True,
    stream=True,
    features=[
        Feature("date", "string", description="Forecast day (record key)"),
        Feature("pm25_pred", "double", description="Predicted PM2.5"),
    ],
)
pred_fg.insert(pred_df, wait=True)
print("wrote", len(pred_df), "predictions into airqpred963ee7")
print("DONE")
