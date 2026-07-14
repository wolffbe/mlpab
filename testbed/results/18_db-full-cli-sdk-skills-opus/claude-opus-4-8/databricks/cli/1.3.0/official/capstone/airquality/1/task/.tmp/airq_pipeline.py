# Databricks notebook source
# FTI pipeline for PM2.5 forecasting — runs entirely on the platform.
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from pyspark.sql import functions as F

CATALOG = "workspace"
SCHEMA = "mlpab2cccc6"
FQ = f"{CATALOG}.{SCHEMA}"
FG = f"{FQ}.airq0ecd46"
TD = f"{FQ}.airqtd0ecd46"
PRED = f"{FQ}.airqpred0ecd46"
ONLINE = f"{FQ}.airqpred0ecd46_online"
MODEL = f"{FQ}.airqmodel0ecd46"
VOL = "/Volumes/workspace/mlpab2cccc6/raw"
USER = spark.sql("select current_user()").first()[0]
EXP = f"/Users/{USER}/mlpab2cccc6/airq_experiment"

results = {}

# COMMAND ----------
# ---- Load raw data ----
hist = (spark.read.option("header", True).option("inferSchema", True)
        .csv(f"{VOL}/airquality_history.csv"))
fc = (spark.read.option("header", True).option("inferSchema", True)
      .csv(f"{VOL}/forecast_days.csv"))
print("hist", hist.count(), "forecast", fc.count())
hist.printSchema()

# COMMAND ----------
# ---- Feature engineering (per-row, available for both history and forecast) ----
def engineer(df):
    d = df.withColumn("date", F.to_date("date"))
    d = (d
         .withColumn("doy", F.dayofyear("date"))
         .withColumn("month", F.month("date"))
         .withColumn("doy_sin", F.sin(2 * np.pi * F.col("doy") / 365.0))
         .withColumn("doy_cos", F.cos(2 * np.pi * F.col("doy") / 365.0))
         .withColumn("temp_hum", F.col("temperature") * F.col("humidity") / 100.0)
         .withColumn("wind_precip", F.col("wind_speed") * (1 + F.col("precipitation"))))
    return d

FEATURES = ["pm25_lag1", "temperature", "humidity", "wind_speed", "pressure",
            "precipitation", "doy", "month", "doy_sin", "doy_cos",
            "temp_hum", "wind_precip"]

hist_f = engineer(hist)
fc_f = engineer(fc)

# Feature group: store engineered features + label, keyed by date.
fg_df = hist_f.select(["date"] + FEATURES + ["pm25"])

from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
fe = FeatureEngineeringClient()

spark.sql(f"DROP TABLE IF EXISTS {FG}")
fe.create_table(
    name=FG,
    primary_keys=["date"],
    df=fg_df,
    description="Engineered air-quality + weather features for PM2.5 forecasting.",
)
print("feature group created:", FG)

# COMMAND ----------
# ---- Assemble training dataset via feature store join ----
label_df = hist_f.select("date", "pm25")
training_set = fe.create_training_set(
    df=label_df,
    feature_lookups=[FeatureLookup(table_name=FG, lookup_key="date", feature_names=FEATURES)],
    label="pm25",
    exclude_columns=[],
)
training_df = training_set.load_df()
spark.sql(f"DROP TABLE IF EXISTS {TD}")
training_df.write.mode("overwrite").saveAsTable(TD)
print("training dataset:", TD, training_df.count())

# COMMAND ----------
# ---- Train regressor (time-ordered holdout) ----
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

pdf = training_df.toPandas().sort_values("date").reset_index(drop=True)
X = pdf[FEATURES].astype(float).values
y = pdf["pm25"].astype(float).values

n = len(pdf)
split = int(n * 0.8)
Xtr, Xval = X[:split], X[split:]
ytr, yval = y[:split], y[split:]

candidates = {
    "gbr": GradientBoostingRegressor(n_estimators=400, max_depth=3, learning_rate=0.05,
                                     subsample=0.9, random_state=42),
    "ridge": Ridge(alpha=1.0),
}
best_name, best_rmse, best_model = None, float("inf"), None
val_metrics = {}
for name, m in candidates.items():
    m.fit(Xtr, ytr)
    pv = m.predict(Xval)
    rmse = float(np.sqrt(mean_squared_error(yval, pv)))
    val_metrics[name] = rmse
    print(f"{name} holdout RMSE = {rmse:.4f}")
    if rmse < best_rmse:
        best_name, best_rmse, best_model = name, rmse, m

print("BEST:", best_name, best_rmse)

# Refit best on ALL history for final model.
from sklearn.base import clone
final_model = clone(best_model)
final_model.fit(X, y)

# Full-data fit metrics + held-out metrics for registration.
val_pred = best_model.predict(Xval)
val_rmse = float(np.sqrt(mean_squared_error(yval, val_pred)))
val_mae = float(mean_absolute_error(yval, val_pred))
val_r2 = float(r2_score(yval, val_pred))
results["holdout_rmse"] = val_rmse
results["holdout_mae"] = val_mae
results["holdout_r2"] = val_r2
results["best_model"] = best_name

# COMMAND ----------
# ---- Register model WITH metrics in Unity Catalog ----
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXP)

from mlflow.models.signature import infer_signature
sig = infer_signature(pdf[FEATURES], final_model.predict(X))

with mlflow.start_run(run_name="airq_pm25") as run:
    mlflow.log_param("model_type", best_name)
    mlflow.log_param("features", ",".join(FEATURES))
    mlflow.log_metric("holdout_rmse", val_rmse)
    mlflow.log_metric("holdout_mae", val_mae)
    mlflow.log_metric("holdout_r2", val_r2)
    for k, v in val_metrics.items():
        mlflow.log_metric(f"val_rmse_{k}", v)
    info = mlflow.sklearn.log_model(
        sk_model=final_model,
        artifact_path="model",
        signature=sig,
        input_example=pdf[FEATURES].head(3),
        registered_model_name=MODEL,
    )
    run_id = run.info.run_id
print("registered model:", MODEL, "holdout_rmse:", val_rmse)

# Tag the registered model version with metrics for visibility.
from mlflow.tracking import MlflowClient
mc = MlflowClient(registry_uri="databricks-uc")
versions = mc.search_model_versions(f"name='{MODEL}'")
latest = max(versions, key=lambda v: int(v.version))
mc.set_model_version_tag(MODEL, latest.version, "holdout_rmse", f"{val_rmse:.4f}")
mc.set_model_version_tag(MODEL, latest.version, "holdout_mae", f"{val_mae:.4f}")
mc.set_model_version_tag(MODEL, latest.version, "holdout_r2", f"{val_r2:.4f}")

# COMMAND ----------
# ---- Predict forecast rows ----
fc_pdf = fc_f.toPandas().sort_values("date").reset_index(drop=True)
Xf = fc_pdf[FEATURES].astype(float).values
fc_pdf["pm25_pred"] = final_model.predict(Xf).astype(float)
pred_out = fc_pdf[["date", "pm25_pred"]].copy()
pred_sdf = spark.createDataFrame(pred_out)
pred_sdf = pred_sdf.withColumn("date", F.to_date("date"))

spark.sql(f"DROP TABLE IF EXISTS {ONLINE}")
spark.sql(f"DROP TABLE IF EXISTS {PRED}")
fe.create_table(
    name=PRED,
    primary_keys=["date"],
    df=pred_sdf,
    description="PM2.5 predictions for forecast days.",
)
spark.sql(f"ALTER TABLE {PRED} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
print("predictions table:", PRED, pred_sdf.count())
display(pred_sdf.orderBy("date"))

# COMMAND ----------
# ---- Online (low-latency) table for predictions ----
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import OnlineTableSpec, OnlineTableSpecTriggeredSchedulingPolicy
w = WorkspaceClient()
online_ok = False
try:
    spec = OnlineTableSpec(
        source_table_full_name=PRED,
        primary_key_columns=["date"],
        run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict({"triggered": "true"}),
    )
    w.online_tables.create(name=ONLINE, spec=spec)
    online_ok = True
    print("online table created:", ONLINE)
except Exception as e:
    print("online_tables create failed:", repr(e))
    # Fallback: publish to online store via feature engineering publish_table
    try:
        from databricks.feature_engineering.online_store_spec import AmazonDynamoDBSpec  # may not exist
    except Exception:
        pass

results["online_ok"] = online_ok

# COMMAND ----------
print("RESULTS:", results)
dbutils.notebook.exit(str(results))
