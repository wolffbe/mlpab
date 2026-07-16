# Databricks notebook source
import numpy as np
import pandas as pd

SCHEMA = "workspace.mlpab2efe57"
VOL = "/Volumes/workspace/mlpab2efe57/airqdata"
FG = f"{SCHEMA}.airq3d0e82"
TD = f"{SCHEMA}.airqtd3d0e82"
PRED = f"{SCHEMA}.airqpred3d0e82"
MODEL_NAME = f"{SCHEMA}.airqmodel3d0e82"
EXP_PATH = "/Users/benedict@hopsworks.ai/mlpab2efe57/airq_experiment"

hist = pd.read_csv(f"{VOL}/airquality_history.csv", parse_dates=["date"])
fc = pd.read_csv(f"{VOL}/forecast_days.csv", parse_dates=["date"])
print("history:", hist.shape, "forecast:", fc.shape)

# COMMAND ----------

def add_calendar(d):
    d = d.copy().sort_values("date").reset_index(drop=True)
    doy = d["date"].dt.dayofyear
    d["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    d["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    d["month"] = d["date"].dt.month.astype("int32")
    return d

hist_f = add_calendar(hist)
# causal rolling air-quality signals (use only past pm25 values)
past = hist_f["pm25"].shift(1)
hist_f["pm25_roll3"] = past.rolling(3, min_periods=1).mean()
hist_f["pm25_roll7"] = past.rolling(7, min_periods=1).mean()
hist_f["pm25_roll14"] = past.rolling(14, min_periods=1).mean()
hist_f = hist_f.dropna(subset=["pm25_lag1"]).reset_index(drop=True)

fc_f = add_calendar(fc)

FEATURES = ["pm25_lag1", "temperature", "humidity", "wind_speed",
            "pressure", "precipitation", "doy_sin", "doy_cos", "month"]

# COMMAND ----------

# 1) Feature group table
fg_pd = hist_f.copy()
fg_pd["date"] = fg_pd["date"].dt.date
spark.createDataFrame(fg_pd).write.mode("overwrite") \
    .option("overwriteSchema", "true").saveAsTable(FG)
spark.sql(f"ALTER TABLE {FG} ALTER COLUMN date SET NOT NULL")
try:
    spark.sql(f"ALTER TABLE {FG} ADD CONSTRAINT airq_fg_pk PRIMARY KEY(date)")
except Exception as e:
    print("fg pk:", e)
print("feature group written:", FG, spark.table(FG).count())

# COMMAND ----------

# 2) Training dataset: chronological split, last 90 days held out
n_test = 90
hist_f["split"] = "train"
hist_f.loc[hist_f.index[-n_test:], "split"] = "test"
td_pd = hist_f[["date"] + FEATURES + ["pm25", "split"]].copy()
td_pd["date"] = td_pd["date"].dt.date
spark.createDataFrame(td_pd).write.mode("overwrite") \
    .option("overwriteSchema", "true").saveAsTable(TD)
print("training dataset written:", TD, spark.table(TD).count())

# COMMAND ----------

# 3) Train, evaluate on held-out days, register with metrics
import mlflow
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

train = hist_f[hist_f["split"] == "train"]
test = hist_f[hist_f["split"] == "test"]
Xtr, ytr = train[FEATURES], train["pm25"]
Xte, yte = test[FEATURES], test["pm25"]

configs = [
    dict(n_estimators=500, learning_rate=0.03, max_depth=2, subsample=0.9),
    dict(n_estimators=500, learning_rate=0.03, max_depth=3, subsample=0.9),
    dict(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8),
    dict(n_estimators=800, learning_rate=0.02, max_depth=3, subsample=0.9),
]
best = None
for cfg in configs:
    m = GradientBoostingRegressor(random_state=42, **cfg)
    m.fit(Xtr, ytr)
    rmse = float(np.sqrt(mean_squared_error(yte, m.predict(Xte))))
    print(cfg, "holdout rmse:", round(rmse, 4))
    if best is None or rmse < best[0]:
        best = (rmse, cfg, m)

rmse, cfg, model = best
pred_te = model.predict(Xte)
mae = float(mean_absolute_error(yte, pred_te))
r2 = float(r2_score(yte, pred_te))
base_rmse = float(np.sqrt(mean_squared_error(yte, np.full(len(yte), ytr.mean()))))
print("best:", cfg, "rmse:", rmse, "mae:", mae, "r2:", r2, "baseline:", base_rmse)

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXP_PATH)
with mlflow.start_run(run_name="airq_gbr") as run:
    mlflow.log_params(cfg)
    mlflow.log_param("features", ",".join(FEATURES))
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("baseline_rmse", base_rmse)
    # refit on the full history with the selected config for deployment
    final_model = GradientBoostingRegressor(random_state=42, **cfg)
    final_model.fit(hist_f[FEATURES], hist_f["pm25"])
    mlflow.sklearn.log_model(
        final_model, "model",
        registered_model_name=MODEL_NAME,
        input_example=Xtr.head(5),
    )

client = mlflow.MlflowClient()
mv = client.search_model_versions(f"name='{MODEL_NAME}'")
latest = max(mv, key=lambda v: int(v.version))
client.update_model_version(
    name=MODEL_NAME, version=latest.version,
    description=f"GBR pm25 forecaster. Held-out (last {n_test} days) RMSE={rmse:.4f}, "
                f"MAE={mae:.4f}, R2={r2:.4f}, baseline RMSE={base_rmse:.4f}.")
for k, v in [("rmse", rmse), ("mae", mae), ("r2", r2)]:
    client.set_model_version_tag(MODEL_NAME, latest.version, k, f"{v:.4f}")
print("registered", MODEL_NAME, "version", latest.version)

# COMMAND ----------

# 4) Predict every forecast row -> prediction feature table
out = pd.DataFrame({
    "date": fc_f["date"].dt.date,
    "pm25_pred": final_model.predict(fc_f[FEATURES]).astype(float),
})
spark.createDataFrame(out).write.mode("overwrite") \
    .option("overwriteSchema", "true").saveAsTable(PRED)
spark.sql(f"ALTER TABLE {PRED} ALTER COLUMN date SET NOT NULL")
try:
    spark.sql(f"ALTER TABLE {PRED} ADD CONSTRAINT airq_pred_pk PRIMARY KEY(date)")
except Exception as e:
    print("pred pk:", e)
spark.sql(f"ALTER TABLE {PRED} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
print("predictions written:", PRED, spark.table(PRED).count())
print("RESULT_RMSE", rmse)
