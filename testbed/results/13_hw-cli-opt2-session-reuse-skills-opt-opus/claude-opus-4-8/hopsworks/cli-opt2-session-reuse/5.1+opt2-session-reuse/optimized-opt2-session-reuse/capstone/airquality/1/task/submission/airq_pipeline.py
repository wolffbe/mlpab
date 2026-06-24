"""FTI training + batch inference, runs as a Hopsworks job (server-side)."""
import os
import numpy as np
import pandas as pd
import joblib
import hopsworks
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

FEATURES = ["pm25_lag1", "temperature", "humidity", "wind_speed",
            "pressure", "precipitation"]

print(">>> logging in", flush=True)
project = hopsworks.login()
fs = project.get_feature_store()

# ---- read training dataset from the feature view (point-in-time snapshot) ----
fv = fs.get_feature_view(name="airqtd06009f", version=1)
X_train, X_test, y_train, y_test = fv.get_train_test_split(training_dataset_version=1)

def prep(df):
    return df[FEATURES].astype(float)

Xtr, Xte = prep(X_train), prep(X_test)
ytr = np.asarray(y_train).ravel().astype(float)
yte = np.asarray(y_test).ravel().astype(float)
print(f">>> train={len(Xtr)} test={len(Xte)}", flush=True)

# ---- train; pick the better of two robust regressors on the held-out split ----
candidates = {
    "hgb": HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05,
                                         max_depth=3, l2_regularization=1.0,
                                         random_state=42),
    "rf": RandomForestRegressor(n_estimators=500, max_depth=None,
                                min_samples_leaf=2, random_state=42, n_jobs=-1),
}
best_name, best_rmse, best_model = None, float("inf"), None
for name, m in candidates.items():
    m.fit(Xtr, ytr)
    rmse = float(np.sqrt(mean_squared_error(yte, m.predict(Xte))))
    print(f">>> {name} test RMSE = {rmse:.4f}", flush=True)
    if rmse < best_rmse:
        best_name, best_rmse, best_model = name, rmse, m

pred_te = best_model.predict(Xte)
metrics = {
    "rmse": best_rmse,
    "mae": float(mean_absolute_error(yte, pred_te)),
    "r2": float(r2_score(yte, pred_te)),
}
print(f">>> BEST={best_name} metrics={metrics}", flush=True)

# ---- register model with metrics ----
from hsml.schema import Schema
from hsml.model_schema import ModelSchema

model_dir = "airqmodel06009f_dir"
os.makedirs(model_dir, exist_ok=True)
joblib.dump(best_model, os.path.join(model_dir, "model.pkl"))

input_schema = Schema(Xtr)
output_schema = Schema(pd.DataFrame({"pm25": ytr}))
model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)

mr = project.get_model_registry()
hops_model = mr.python.create_model(
    name="airqmodel06009f",
    metrics=metrics,
    model_schema=model_schema,
    input_example=Xtr.head(1),
    description=f"PM2.5 daily regressor ({best_name}); held-out RMSE={best_rmse:.4f}",
)
hops_model.save(model_dir)
print(">>> model registered", flush=True)

# ---- batch inference over forecast rows ----
fcst = fs.get_feature_group(name="airqforecast06009f", version=1).read()
fcst = fcst.sort_values("date").reset_index(drop=True)
Xf = fcst[FEATURES].astype(float)
preds = best_model.predict(Xf)
pred_df = pd.DataFrame({"date": fcst["date"].astype(str),
                        "pm25_pred": np.asarray(preds, dtype=float)})
print(f">>> predicted {len(pred_df)} rows", flush=True)
print(pred_df.head().to_string(), flush=True)

# ---- write predictions to an online+offline feature table ----
pred_fg = fs.get_or_create_feature_group(
    name="airqpred06009f",
    version=1,
    primary_key=["date"],
    description="PM2.5 predictions for forecast days",
    online_enabled=True,
)
pred_fg.insert(pred_df)
print(">>> predictions written to airqpred06009f", flush=True)
print(">>> DONE", flush=True)
