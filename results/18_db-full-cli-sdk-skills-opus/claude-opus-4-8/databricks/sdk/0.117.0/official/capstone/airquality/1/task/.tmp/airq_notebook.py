# Databricks notebook source
# MAGIC %pip install mlflow scikit-learn --quiet
# COMMAND ----------
dbutils.library.restartPython()
# COMMAND ----------
# Full FTI pipeline for PM2.5 air-quality forecasting.

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

CAT = "workspace"
SCHEMA = "mlpab600b4f"
FQ = f"{CAT}.{SCHEMA}"
VOL = f"/Volumes/{CAT}/{SCHEMA}/airqdata"
USER = spark.sql("select current_user()").collect()[0][0]
EXP = f"/Users/{USER}/mlpab600b4f/airq_experiment"

FG = f"{FQ}.airq0ecd46"
TD = f"{FQ}.airqtd0ecd46"
PRED = f"{FQ}.airqpred0ecd46"
MODEL = f"{FQ}.airqmodel0ecd46"

BASE_FEATS = ["pm25_lag1", "temperature", "humidity", "wind_speed", "pressure", "precipitation"]

# ---------- Load raw data ----------
hist = pd.read_csv(f"{VOL}/airquality_history.csv")
fc = pd.read_csv(f"{VOL}/forecast_days.csv")

def engineer(df):
    df = df.copy()
    d = pd.to_datetime(df["date"])
    doy = d.dt.dayofyear.astype(float)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["month"] = d.dt.month.astype(float)
    return df

hist_fe = engineer(hist)
fc_fe = engineer(fc)
FEATS = BASE_FEATS + ["doy_sin", "doy_cos", "month"]

# ---------- 1. Feature group ----------
fg_cols = ["date"] + FEATS + ["pm25"]
fg_pdf = hist_fe[fg_cols]
schema = StructType([StructField("date", StringType(), False)] +
                    [StructField(c, DoubleType(), True) for c in FEATS + ["pm25"]])
fg_sdf = spark.createDataFrame(fg_pdf.astype({c: float for c in FEATS + ["pm25"]}), schema=schema)
fg_sdf.write.mode("overwrite").option("overwriteSchema", "true").format("delta").saveAsTable(FG)
spark.sql(f"ALTER TABLE {FG} ALTER COLUMN date SET NOT NULL")
try:
    spark.sql(f"ALTER TABLE {FG} ADD CONSTRAINT pk_airqfg PRIMARY KEY (date)")
except Exception as e:
    print("fg pk:", e)
print("feature group rows:", spark.table(FG).count())

# ---------- 2. Training dataset ----------
td_sdf = spark.table(FG)
td_sdf.write.mode("overwrite").option("overwriteSchema", "true").format("delta").saveAsTable(TD)
print("training dataset rows:", spark.table(TD).count())

# ---------- 3. Train + register model ----------
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXP)

train_pdf = spark.table(TD).toPandas().sort_values("date").reset_index(drop=True)
X = train_pdf[FEATS]
y = train_pdf["pm25"].astype(float)

# Time-based held-out split for honest metric estimate
n = len(train_pdf)
cut = int(n * 0.8)
Xtr, Xval = X.iloc[:cut], X.iloc[cut:]
ytr, yval = y.iloc[:cut], y.iloc[cut:]

params = dict(n_estimators=500, learning_rate=0.03, max_depth=3, subsample=0.9, random_state=42)
val_model = GradientBoostingRegressor(**params).fit(Xtr, ytr)
val_pred = val_model.predict(Xval)
rmse = float(np.sqrt(mean_squared_error(yval, val_pred)))
print("held-out RMSE:", rmse)

# Final model on all data
final_model = GradientBoostingRegressor(**params).fit(X, y)
full_rmse = float(np.sqrt(mean_squared_error(y, final_model.predict(X))))

with mlflow.start_run(run_name="airq_gbr") as run:
    mlflow.log_params(params)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("holdout_rmse", rmse)
    mlflow.log_metric("train_rmse", full_rmse)
    sig = mlflow.models.infer_signature(X, final_model.predict(X))
    mlflow.sklearn.log_model(
        final_model, artifact_path="model",
        registered_model_name=MODEL,
        signature=sig, input_example=X.head(3))
    run_id = run.info.run_id
print("registered model:", MODEL, "run:", run_id)

# ---------- 4. Predict forecast -> predictions feature table ----------
fc_pred = fc_fe.copy()
fc_pred["pm25_pred"] = final_model.predict(fc_fe[FEATS]).astype(float)
pred_pdf = fc_pred[["date", "pm25_pred"]]
pred_schema = StructType([StructField("date", StringType(), False),
                          StructField("pm25_pred", DoubleType(), True)])
pred_sdf = spark.createDataFrame(pred_pdf.astype({"pm25_pred": float}), schema=pred_schema)
pred_sdf.write.mode("overwrite").option("overwriteSchema", "true").format("delta") \
    .option("delta.enableChangeDataFeed", "true").saveAsTable(PRED)
spark.sql(f"ALTER TABLE {PRED} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
spark.sql(f"ALTER TABLE {PRED} ALTER COLUMN date SET NOT NULL")
try:
    spark.sql(f"ALTER TABLE {PRED} ADD CONSTRAINT pk_airqpred PRIMARY KEY (date)")
except Exception as e:
    print("pred pk:", e)
print("prediction rows:", spark.table(PRED).count())

print("DONE rmse=%s" % rmse)
