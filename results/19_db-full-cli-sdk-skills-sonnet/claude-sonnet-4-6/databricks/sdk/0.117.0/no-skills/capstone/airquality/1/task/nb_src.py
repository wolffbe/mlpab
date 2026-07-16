import json, math, mlflow, mlflow.sklearn
import numpy as np
from sklearn.ensemble import (
    GradientBoostingRegressor, RandomForestRegressor,
    ExtraTreesRegressor, HistGradientBoostingRegressor
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window

spark = SparkSession.builder.getOrCreate()

# Parameters (injected via notebook params)
catalog       = dbutils.widgets.get("catalog")
db_schema     = dbutils.widgets.get("db_schema")
vol_path      = dbutils.widgets.get("vol_path")
user          = dbutils.widgets.get("user")
prefix        = dbutils.widgets.get("prefix")

cs = f"{catalog}.{db_schema}"
fg_name    = "airqfdfb59"
td_name    = "airqtdfdfb59"
model_name = "airqmodelfdfb59"
pred_name  = "airqpredfdfb59"

print(f"catalog.schema = {cs}")

# ── Load data ──────────────────────────────────────────────────────────────
hist_df = (spark.read.format("csv").option("header","true").option("inferSchema","true")
           .load(f"{vol_path}/airquality_history.csv")
           .withColumn("date", F.to_date("date"))
           .withColumn("is_forecast", F.lit(0)))

fc_df = (spark.read.format("csv").option("header","true").option("inferSchema","true")
         .load(f"{vol_path}/forecast_days.csv")
         .withColumn("date", F.to_date("date"))
         .withColumn("pm25", F.lit(None).cast("double"))
         .withColumn("is_forecast", F.lit(1)))

print(f"History: {hist_df.count()} rows, Forecast: {fc_df.count()} rows")

# Combine history + forecast for proper rolling features
# Need to fill in missing forecast dates for correct temporal ordering
all_df = hist_df.unionByName(fc_df).orderBy("date")
print(f"Combined rows: {all_df.count()}")

# ── Feature Engineering on Combined Data ─────────────────────────────────
w3  = Window.orderBy(F.unix_date(F.col("date"))).rowsBetween(-2, 0)
w7  = Window.orderBy(F.unix_date(F.col("date"))).rowsBetween(-6, 0)
w14 = Window.orderBy(F.unix_date(F.col("date"))).rowsBetween(-13, 0)
w30 = Window.orderBy(F.unix_date(F.col("date"))).rowsBetween(-29, 0)

def engineer(df):
    df = df.withColumn("pm25_lag1_v", F.coalesce("pm25_lag1", F.lit(10.0)))
    # Rolling means of pm25 lag
    df = df.withColumn("pm25_roll3",    F.avg("pm25_lag1_v").over(w3))
    df = df.withColumn("pm25_roll7",    F.avg("pm25_lag1_v").over(w7))
    df = df.withColumn("pm25_roll14",   F.avg("pm25_lag1_v").over(w14))
    df = df.withColumn("pm25_roll30",   F.avg("pm25_lag1_v").over(w30))
    # Rolling std
    df = df.withColumn("pm25_std7",     F.stddev("pm25_lag1_v").over(w7))
    df = df.withColumn("pm25_std14",    F.stddev("pm25_lag1_v").over(w14))
    # Rolling weather
    df = df.withColumn("temp_roll7",    F.avg("temperature").over(w7))
    df = df.withColumn("hum_roll7",     F.avg("humidity").over(w7))
    df = df.withColumn("wind_roll7",    F.avg("wind_speed").over(w7))
    df = df.withColumn("precip_roll3",  F.sum("precipitation").over(w3))
    df = df.withColumn("precip_roll7",  F.sum("precipitation").over(w7))
    df = df.withColumn("temp_roll14",   F.avg("temperature").over(w14))
    df = df.withColumn("wind_roll14",   F.avg("wind_speed").over(w14))
    df = df.withColumn("hum_roll14",    F.avg("humidity").over(w14))
    # Derived
    df = df.withColumn("temp_hum",      F.col("temperature") * F.col("humidity") / 100.0)
    df = df.withColumn("wind_pres",     F.col("wind_speed") * F.col("pressure") / 1000.0)
    df = df.withColumn("inv_wind",      F.lit(1.0) / (F.col("wind_speed") + 0.1))
    df = df.withColumn("high_hum",      F.when(F.col("humidity") > 80, 1.0).otherwise(0.0))
    df = df.withColumn("low_wind",      F.when(F.col("wind_speed") < 5, 1.0).otherwise(0.0))
    df = df.withColumn("pm25_sq",       F.col("pm25_lag1_v") * F.col("pm25_lag1_v"))
    df = df.withColumn("pm25_wind",     F.col("pm25_lag1_v") * F.col("inv_wind"))
    df = df.withColumn("pm25_ratio",    F.col("pm25_roll3") / (F.col("pm25_roll14") + 0.1))
    # Seasonal
    df = df.withColumn("doy",           F.dayofyear("date"))
    df = df.withColumn("month",         F.month("date"))
    df = df.withColumn("quarter",       F.quarter("date"))
    df = df.withColumn("sin_doy",       F.sin(2 * math.pi * F.col("doy") / 365.0))
    df = df.withColumn("cos_doy",       F.cos(2 * math.pi * F.col("doy") / 365.0))
    df = df.withColumn("sin_doy2",      F.sin(4 * math.pi * F.col("doy") / 365.0))
    df = df.withColumn("cos_doy2",      F.cos(4 * math.pi * F.col("doy") / 365.0))
    # Fill NaN std (first rows)
    df = df.withColumn("pm25_std7",     F.coalesce("pm25_std7",  F.lit(0.0)))
    df = df.withColumn("pm25_std14",    F.coalesce("pm25_std14", F.lit(0.0)))
    return df

all_feat = engineer(all_df)

# Split back
hist_feat = all_feat.filter(F.col("is_forecast") == 0)
fc_feat   = all_feat.filter(F.col("is_forecast") == 1)
print(f"After split - hist: {hist_feat.count()}, fc: {fc_feat.count()}")

# ── Write Feature Group ────────────────────────────────────────────────────
fg_table = f"{cs}.{fg_name}"
spark.sql(f"DROP TABLE IF EXISTS {fg_table}")
(hist_feat.withColumn("date", F.col("date").cast("string"))
 .drop("is_forecast")
 .write.format("delta").mode("overwrite").saveAsTable(fg_table))
print(f"Feature group: {fg_table}")

# ── Training Dataset ───────────────────────────────────────────────────────
feat_cols = [
    "pm25_lag1", "temperature", "humidity", "wind_speed", "pressure", "precipitation",
    "pm25_roll3", "pm25_roll7", "pm25_roll14", "pm25_roll30",
    "pm25_std7", "pm25_std14",
    "temp_roll7", "hum_roll7", "wind_roll7", "precip_roll3", "precip_roll7",
    "temp_roll14", "wind_roll14", "hum_roll14",
    "temp_hum", "wind_pres", "inv_wind", "high_hum", "low_wind",
    "pm25_sq", "pm25_wind", "pm25_ratio",
    "sin_doy", "cos_doy", "sin_doy2", "cos_doy2", "month", "quarter"
]
td_df = hist_feat.select(["date"] + feat_cols + ["pm25"]).dropna(subset=["pm25"])

td_table = f"{cs}.{td_name}"
spark.sql(f"DROP TABLE IF EXISTS {td_table}")
(td_df.withColumn("date", F.col("date").cast("string"))
 .write.format("delta").mode("overwrite").saveAsTable(td_table))
print(f"Training dataset: {td_table}")

# ── Collect to Pandas ─────────────────────────────────────────────────────
td_pd = td_df.orderBy("date").toPandas()
print(f"Pandas shape: {td_pd.shape}")

for c in feat_cols:
    td_pd[c] = td_pd[c].fillna(td_pd[c].median())

# Train/test split: last 90 rows as test
n_test = 90
train_pd = td_pd.iloc[:-n_test]
test_pd  = td_pd.iloc[-n_test:]
print(f"Train: {len(train_pd)}, Test: {len(test_pd)}")

X_train = train_pd[feat_cols].values
y_train = train_pd["pm25"].values
X_test  = test_pd[feat_cols].values
y_test  = test_pd["pm25"].values
X_all   = td_pd[feat_cols].values
y_all   = td_pd["pm25"].values

# ── Train models ────────────────────────────────────────────────────────────
model_preds = {}

# 1. HistGradientBoosting
hgb = HistGradientBoostingRegressor(
    max_iter=500, max_depth=5, learning_rate=0.03,
    min_samples_leaf=8, l2_regularization=0.1, random_state=42
)
hgb.fit(X_train, y_train)
p_hgb = hgb.predict(X_test)
rmse_hgb = float(np.sqrt(mean_squared_error(y_test, p_hgb)))
print(f"HGB RMSE={rmse_hgb:.4f}")
model_preds["HGB"] = (hgb, p_hgb, rmse_hgb)

# 2. GBR
gbr = GradientBoostingRegressor(
    n_estimators=500, max_depth=4, learning_rate=0.03,
    subsample=0.8, min_samples_leaf=5, max_features=0.8, random_state=42
)
gbr.fit(X_train, y_train)
p_gbr = gbr.predict(X_test)
rmse_gbr = float(np.sqrt(mean_squared_error(y_test, p_gbr)))
print(f"GBR RMSE={rmse_gbr:.4f}")
model_preds["GBR"] = (gbr, p_gbr, rmse_gbr)

# 3. RF
rfr = RandomForestRegressor(
    n_estimators=500, max_depth=None, min_samples_leaf=3,
    max_features=0.7, random_state=42, n_jobs=-1
)
rfr.fit(X_train, y_train)
p_rf = rfr.predict(X_test)
rmse_rf = float(np.sqrt(mean_squared_error(y_test, p_rf)))
print(f"RF  RMSE={rmse_rf:.4f}")
model_preds["RF"] = (rfr, p_rf, rmse_rf)

# 4. ET
etr = ExtraTreesRegressor(
    n_estimators=500, max_depth=None, min_samples_leaf=2,
    max_features=0.7, random_state=42, n_jobs=-1
)
etr.fit(X_train, y_train)
p_et = etr.predict(X_test)
rmse_et = float(np.sqrt(mean_squared_error(y_test, p_et)))
print(f"ET  RMSE={rmse_et:.4f}")
model_preds["ET"] = (etr, p_et, rmse_et)

# Try XGBoost
try:
    import xgboost as xgb
    xgbr = xgb.XGBRegressor(
        n_estimators=500, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1, verbosity=0
    )
    xgbr.fit(X_train, y_train)
    p_xgb = xgbr.predict(X_test)
    rmse_xgb = float(np.sqrt(mean_squared_error(y_test, p_xgb)))
    print(f"XGB RMSE={rmse_xgb:.4f}")
    model_preds["XGB"] = (xgbr, p_xgb, rmse_xgb)
except ImportError:
    print("XGBoost not available")

# Equal ensemble
names = list(model_preds.keys())
preds_stack = np.stack([model_preds[k][1] for k in names])
p_equal = preds_stack.mean(axis=0)
rmse_equal = float(np.sqrt(mean_squared_error(y_test, p_equal)))
print(f"Equal Ensemble RMSE={rmse_equal:.4f}")

# Inverse-RMSE weighted ensemble
rmses = np.array([model_preds[k][2] for k in names])
w_inv = (1.0 / rmses) / (1.0 / rmses).sum()
p_weighted = (preds_stack * w_inv[:, None]).sum(axis=0)
rmse_weighted = float(np.sqrt(mean_squared_error(y_test, p_weighted)))
print(f"Weighted Ensemble RMSE={rmse_weighted:.4f}")

# Pick best
all_options = {**{k: (v[1], v[2]) for k, v in model_preds.items()},
               "EqualEns": (p_equal, rmse_equal),
               "WeightedEns": (p_weighted, rmse_weighted)}
best_name = min(all_options, key=lambda k: all_options[k][1])
best_preds, rmse = all_options[best_name]
mae = float(mean_absolute_error(y_test, best_preds))
r2  = float(r2_score(y_test, best_preds))
print(f"Best: {best_name}, RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

# ── Retrain on ALL data ────────────────────────────────────────────────────
for nm, (mod, _, _) in model_preds.items():
    mod.fit(X_all, y_all)
    print(f"Retrained {nm} on {len(X_all)} rows")

# ── Predict Forecast ──────────────────────────────────────────────────────
fc_pd = fc_feat.orderBy("date").toPandas()
for c in feat_cols:
    fc_pd[c] = fc_pd[c].fillna(fc_pd[c].median() if not fc_pd[c].isna().all() else 10.0)
X_fc = fc_pd[feat_cols].values

fc_stack = np.stack([model_preds[k][0].predict(X_fc) for k in names])
# Use same approach as best validation
if best_name == "EqualEns":
    y_fc = fc_stack.mean(axis=0)
elif best_name == "WeightedEns":
    y_fc = (fc_stack * w_inv[:, None]).sum(axis=0)
else:
    y_fc = model_preds[best_name][0].predict(X_fc)

# ── MLflow Register ────────────────────────────────────────────────────────
# Register HGB as the representative model (or best single)
best_single = min(model_preds, key=lambda k: model_preds[k][2])
final_model = model_preds[best_single][0]  # already retrained on all data

mlflow.set_registry_uri("databricks-uc")
exp_path = f"/Users/{user}/{prefix}/airquality_exp"
mlflow.set_experiment(exp_path)

with mlflow.start_run(run_name="pm25_v4") as run:
    mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
    mlflow.log_params({
        "best": best_name, "final_registered": best_single,
        "n_features": len(feat_cols),
        **{f"rmse_{k}": v[2] for k, v in model_preds.items()},
        "rmse_equal_ens": rmse_equal,
        "rmse_weighted_ens": rmse_weighted
    })
    signature = mlflow.models.infer_signature(X_train, y_train)
    mlflow.sklearn.log_model(
        final_model, "model",
        signature=signature,
        registered_model_name=f"{cs}.{model_name}"
    )
    run_id = run.info.run_id
    print(f"Run ID: {run_id}")

print(f"Model registered: {cs}.{model_name}")

# ── Write Predictions ──────────────────────────────────────────────────────
import pandas as pd
pred_pd = pd.DataFrame({
    "date": fc_pd["date"].astype(str),
    "pm25_pred": y_fc.astype(float)
})
print("Predictions (first 10):")
print(pred_pd.head(10))

pred_spark = spark.createDataFrame(pred_pd)

pred_table = f"{cs}.{pred_name}"
spark.sql(f"DROP TABLE IF EXISTS {pred_table}")
(pred_spark.write.format("delta").mode("overwrite")
 .option("overwriteSchema","true")
 .saveAsTable(pred_table))

spark.sql(f"ALTER TABLE {pred_table} SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')")
print(f"Predictions written: {pred_table}")

print("PIPELINE COMPLETE")
result = {"rmse": rmse, "mae": mae, "r2": r2, "best_model": best_name,
          "feature_group": fg_table, "training_dataset": td_table,
          "model": f"{cs}.{model_name}", "predictions": pred_table}
print(json.dumps(result))
dbutils.notebook.exit(json.dumps(result))
