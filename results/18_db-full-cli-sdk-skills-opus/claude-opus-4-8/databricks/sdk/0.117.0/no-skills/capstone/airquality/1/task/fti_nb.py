# Databricks notebook source
# MAGIC %pip install -q mlflow scikit-learn databricks-feature-engineering

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from pyspark.sql import functions as F, Window
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

CAT = "workspace"
SCH = "mlpab23ab6a"
USER = "benedict@logicalclocks.com"
PREFIX = "mlpab23ab6a"
FG = f"{CAT}.{SCH}.airq0ecd46"
TD = f"{CAT}.{SCH}.airqtd0ecd46"
PRED = f"{CAT}.{SCH}.airqpred0ecd46"
MODEL = f"{CAT}.{SCH}.airqmodel0ecd46"
VOL = f"/Volumes/{CAT}/{SCH}/airq_data"

spark.sql(f"USE CATALOG {CAT}")
spark.sql(f"USE SCHEMA {SCH}")

fe = FeatureEngineeringClient()

# ---------- 1. Read raw data ----------
hist = (spark.read.csv(f"{VOL}/airquality_history.csv", header=True, inferSchema=True))
fc = (spark.read.csv(f"{VOL}/forecast_days.csv", header=True, inferSchema=True))

# ---------- 2. Feature engineering (weather + lag/rolling air-quality signals) ----------
def add_calendar(df):
    d = F.to_date(F.col("date"))
    doy = F.dayofyear(d)
    return (df
            .withColumn("month", F.month(d))
            .withColumn("doy_sin", F.sin(2 * np.pi * doy / 365.0))
            .withColumn("doy_cos", F.cos(2 * np.pi * doy / 365.0)))

wspec = Window.orderBy("date")
hist_fe = add_calendar(hist)
hist_fe = (hist_fe
           .withColumn("pm25_roll3", F.avg("pm25").over(wspec.rowsBetween(-3, -1)))
           .withColumn("pm25_roll7", F.avg("pm25").over(wspec.rowsBetween(-7, -1)))
           .withColumn("pm25_std7", F.stddev("pm25").over(wspec.rowsBetween(-7, -1))))

FEAT_COLS = ["date", "pm25_lag1", "temperature", "humidity", "wind_speed",
             "pressure", "precipitation", "month", "doy_sin", "doy_cos",
             "pm25_roll3", "pm25_roll7", "pm25_std7"]
feat_sdf = hist_fe.select(*FEAT_COLS)

# ---------- 1. Feature group ----------
try:
    spark.sql(f"DROP TABLE IF EXISTS {FG}")
except Exception as e:
    print("drop fg", e)
fe.create_table(
    name=FG,
    primary_keys=["date"],
    df=feat_sdf,
    description="Air-quality engineered features (weather + lag/rolling pm25 signals).",
)

# ---------- 2. Training dataset via feature lookup ----------
label_sdf = hist.select("date", "pm25")
lookups = [FeatureLookup(table_name=FG, lookup_key="date")]
training_set = fe.create_training_set(
    df=label_sdf, feature_lookups=lookups, label="pm25", exclude_columns=[])
train_sdf = training_set.load_df()
train_sdf.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(TD)
pdf = spark.table(TD).toPandas().sort_values("date").reset_index(drop=True)

# Model features must be available at inference time (forecast rows have no pm25 history)
MODEL_FEATS = ["pm25_lag1", "temperature", "humidity", "wind_speed",
               "pressure", "precipitation", "month", "doy_sin", "doy_cos"]

X = pdf[MODEL_FEATS].astype(float).fillna(0.0).values
y = pdf["pm25"].astype(float).values
n = len(pdf)
k = int(n * 0.8)
Xtr, Xv = X[:k], X[k:]
ytr, yv = y[:k], y[k:]

baseline_rmse = float(mean_squared_error(yv, np.full_like(yv, ytr.mean())) ** 0.5)

cands = {
    "ridge": Ridge(alpha=1.0),
    "rf": RandomForestRegressor(n_estimators=500, random_state=0, n_jobs=-1),
    "gbr": GradientBoostingRegressor(n_estimators=400, max_depth=3,
                                     learning_rate=0.05, random_state=0),
}
results = {}
best_name, best_rmse, best_model = None, 1e18, None
for name, m in cands.items():
    m.fit(Xtr, ytr)
    p = m.predict(Xv)
    rmse = float(mean_squared_error(yv, p) ** 0.5)
    results[name] = rmse
    if rmse < best_rmse:
        best_name, best_rmse, best_model = name, rmse, m

# ---------- 3. Train final model on all history & register WITH metrics ----------
final_model = cands[best_name]
final_model.fit(X, y)

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(f"/Users/{USER}/{PREFIX}/airq_experiment")
with mlflow.start_run(run_name="airq_train") as run:
    mlflow.log_param("model_type", best_name)
    mlflow.log_param("features", ",".join(MODEL_FEATS))
    for nm, rm in results.items():
        mlflow.log_metric(f"val_rmse_{nm}", rm)
    mlflow.log_metric("val_rmse", best_rmse)
    mlflow.log_metric("baseline_rmse", baseline_rmse)
    sig = infer_signature(pdf[MODEL_FEATS].astype(float).fillna(0.0), final_model.predict(X))
    mlflow.sklearn.log_model(
        final_model, artifact_path="model",
        registered_model_name=MODEL,
        signature=sig,
        input_example=pdf[MODEL_FEATS].astype(float).fillna(0.0).head(3),
    )
    run_id = run.info.run_id

# ---------- 4. Predict forecast days ----------
fc_fe = add_calendar(fc)
fc_pdf = fc_fe.select("date", *MODEL_FEATS).toPandas().sort_values("date").reset_index(drop=True)
Xf = fc_pdf[MODEL_FEATS].astype(float).fillna(0.0).values
preds = final_model.predict(Xf)
pred_pdf = pd.DataFrame({"date": fc_pdf["date"].astype(str).values,
                         "pm25_pred": preds.astype(float)})
pred_sdf = spark.createDataFrame(pred_pdf)

try:
    spark.sql(f"DROP TABLE IF EXISTS {PRED}")
except Exception as e:
    print("drop pred", e)
fe.create_table(
    name=PRED,
    primary_keys=["date"],
    df=pred_sdf,
    description="PM2.5 predictions for forecast days (record key=date, pm25_pred).",
)

summary = {
    "best_model": best_name,
    "val_rmse": best_rmse,
    "baseline_rmse": baseline_rmse,
    "all_val_rmse": results,
    "n_train": n,
    "n_pred": int(pred_pdf.shape[0]),
    "run_id": run_id,
    "model": MODEL,
    "fg": FG, "td": TD, "pred": PRED,
}
print(json.dumps(summary, indent=2))
dbutils.notebook.exit(json.dumps(summary))
