# Databricks notebook source
# FTI pipeline for PM2.5 forecasting

CATALOG = "workspace"
SCHEMA = "mlpab7bc766"
VOL = f"/Volumes/{CATALOG}/{SCHEMA}/airq_raw"
FG = f"{CATALOG}.{SCHEMA}.airq0ecd46"
TD = f"{CATALOG}.{SCHEMA}.airqtd0ecd46"
PRED = f"{CATALOG}.{SCHEMA}.airqpred0ecd46"
MODEL = f"{CATALOG}.{SCHEMA}.airqmodel0ecd46"
EXPERIMENT = "/Users/benedict@logicalclocks.com/mlpab7bc766/airq_experiment"

import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
import mlflow
import mlflow.sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from databricks.feature_engineering import FeatureEngineeringClient

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

# COMMAND ----------
# ---- 1. Read raw data ----
hist = spark.read.csv(f"{VOL}/airquality_history.csv", header=True, inferSchema=True)
fc = spark.read.csv(f"{VOL}/forecast_days.csv", header=True, inferSchema=True)

FEATURE_COLS = ["pm25_lag1", "temperature", "humidity", "wind_speed", "pressure", "precipitation"]

def engineer(df):
    df = df.withColumn("date", F.to_date("date"))
    df = df.withColumn("month", F.month("date").cast(DoubleType()))
    df = df.withColumn("doy", F.dayofyear("date").cast(DoubleType()))
    df = df.withColumn("doy_sin", F.sin(F.col("doy") * 2.0 * np.pi / 365.0))
    df = df.withColumn("doy_cos", F.cos(F.col("doy") * 2.0 * np.pi / 365.0))
    df = df.withColumn("temp_hum", F.col("temperature") * F.col("humidity") / 100.0)
    df = df.withColumn("wind_press", F.col("wind_speed") * F.col("pressure") / 1000.0)
    for c in FEATURE_COLS:
        df = df.withColumn(c, F.col(c).cast(DoubleType()))
    return df

ENG_COLS = FEATURE_COLS + ["month", "doy_sin", "doy_cos", "temp_hum", "wind_press"]

hist_e = engineer(hist)
fc_e = engineer(fc)

# COMMAND ----------
# ---- 2. Feature group airq0ecd46 (keyed by date, includes target pm25) ----
fe = FeatureEngineeringClient()
for t in [FG]:
    try:
        spark.sql(f"DROP TABLE IF EXISTS {t}")
    except Exception as e:
        print(e)

fg_df = hist_e.select(["date"] + ENG_COLS + ["pm25"])
fe.create_table(
    name=FG,
    primary_keys=["date"],
    df=fg_df,
    description="Engineered air-quality + weather features keyed by date, with pm25 target.",
)
print("created feature group", FG)

# COMMAND ----------
# ---- 3. Training dataset airqtd0ecd46 ----
td_pdf = fg_df.toPandas().dropna().sort_values("date").reset_index(drop=True)
spark.createDataFrame(td_pdf).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(TD)
print("training dataset rows:", len(td_pdf))

X = td_pdf[ENG_COLS].astype(float).values
y = td_pdf["pm25"].astype(float).values

# COMMAND ----------
# ---- 4. Train + register model with metrics ----
# time-ordered holdout for honest RMSE
n = len(X)
cut = int(n * 0.8)
Xtr, Xte = X[:cut], X[cut:]
ytr, yte = y[:cut], y[cut:]

def make_model():
    return HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.05, max_depth=4,
        min_samples_leaf=15, l2_regularization=1.0, random_state=42,
    )

m_eval = make_model().fit(Xtr, ytr)
pred_te = m_eval.predict(Xte)
holdout_rmse = float(np.sqrt(mean_squared_error(yte, pred_te)))
baseline_rmse = float(np.sqrt(mean_squared_error(yte, np.full_like(yte, ytr.mean()))))
print("holdout RMSE:", holdout_rmse, "baseline:", baseline_rmse)

# final model on all data
final_model = make_model().fit(X, y)
train_rmse = float(np.sqrt(mean_squared_error(y, final_model.predict(X))))

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT)
with mlflow.start_run(run_name="airq_hgb") as run:
    mlflow.log_params({"max_iter": 500, "learning_rate": 0.05, "max_depth": 4,
                       "min_samples_leaf": 15, "l2_regularization": 1.0})
    mlflow.log_metric("holdout_rmse", holdout_rmse)
    mlflow.log_metric("baseline_rmse", baseline_rmse)
    mlflow.log_metric("train_rmse", train_rmse)
    signature = mlflow.models.infer_signature(td_pdf[ENG_COLS], final_model.predict(X))
    mlflow.sklearn.log_model(
        final_model, artifact_path="model",
        registered_model_name=MODEL,
        signature=signature,
        input_example=td_pdf[ENG_COLS].head(3),
    )
    run_id = run.info.run_id
print("registered model", MODEL, "run", run_id, "holdout_rmse", holdout_rmse)

# COMMAND ----------
# ---- 5. Predict forecast rows -> feature table airqpred0ecd46 ----
fc_pdf = fc_e.select(["date"] + ENG_COLS).toPandas().sort_values("date").reset_index(drop=True)
Xf = fc_pdf[ENG_COLS].astype(float).values
fc_pdf["pm25_pred"] = final_model.predict(Xf).astype(float)

pred_sdf = spark.createDataFrame(fc_pdf[["date", "pm25_pred"]])
try:
    spark.sql(f"DROP TABLE IF EXISTS {PRED}")
except Exception as e:
    print(e)

fe.create_table(
    name=PRED,
    primary_keys=["date"],
    df=pred_sdf,
    description="PM2.5 predictions for forecast days, keyed by date.",
)
print("created prediction feature table", PRED, "rows:", fc_pdf.shape[0])
display(fc_pdf[["date", "pm25_pred"]])
