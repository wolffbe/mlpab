# Databricks notebook source
# Full FTI pipeline: features -> training dataset -> train/register model -> predictions
import pyspark.sql.functions as F
from pyspark.sql.window import Window

CATALOG_SCHEMA = "workspace.mlpab3e3ee5"
FG_TABLE = f"{CATALOG_SCHEMA}.airq3d0e82"
TD_TABLE = f"{CATALOG_SCHEMA}.airqtd3d0e82"
PRED_TABLE = f"{CATALOG_SCHEMA}.airqpred3d0e82"
MODEL_NAME = f"{CATALOG_SCHEMA}.airqmodel3d0e82"
VOL = "/Volumes/workspace/mlpab3e3ee5/data"

# COMMAND ----------
# ---- 1. Feature engineering into feature group airq3d0e82 ----
hist = (spark.read.option("header", True).option("inferSchema", True)
        .csv(f"{VOL}/airquality_history.csv")
        .withColumn("date", F.to_date("date")))
fcst = (spark.read.option("header", True).option("inferSchema", True)
        .csv(f"{VOL}/forecast_days.csv")
        .withColumn("date", F.to_date("date")))

w3 = Window.orderBy("date").rowsBetween(-2, 0)
w7 = Window.orderBy("date").rowsBetween(-6, 0)

def add_features(df):
    doy = F.dayofyear("date")
    return (df
            .withColumn("month", F.month("date"))
            .withColumn("doy_sin", F.sin(2 * 3.141592653589793 * doy / 365.25))
            .withColumn("doy_cos", F.cos(2 * 3.141592653589793 * doy / 365.25)))

hist_feat = (add_features(hist)
             .withColumn("pm25_roll3_mean", F.avg("pm25_lag1").over(w3))
             .withColumn("pm25_roll7_mean", F.avg("pm25_lag1").over(w7))
             .withColumn("pm25_roll7_std", F.coalesce(F.stddev("pm25_lag1").over(w7), F.lit(0.0))))

hist_feat.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(FG_TABLE)
spark.sql(f"ALTER TABLE {FG_TABLE} ALTER COLUMN date SET NOT NULL")
spark.sql(f"ALTER TABLE {FG_TABLE} ADD CONSTRAINT airq3d0e82_pk PRIMARY KEY(date)")
spark.sql(f"COMMENT ON TABLE {FG_TABLE} IS 'Feature group: weather + lag/rolling PM2.5 signals'")
print("feature group rows:", spark.table(FG_TABLE).count())

# COMMAND ----------
# ---- 2. Training dataset airqtd3d0e82 (features usable at forecast time + label) ----
FEATURES = ["pm25_lag1", "temperature", "humidity", "wind_speed", "pressure",
            "precipitation", "doy_sin", "doy_cos"]
td = spark.table(FG_TABLE).select("date", *FEATURES, "pm25").orderBy("date")
td.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(TD_TABLE)
spark.sql(f"ALTER TABLE {TD_TABLE} ALTER COLUMN date SET NOT NULL")
spark.sql(f"ALTER TABLE {TD_TABLE} ADD CONSTRAINT airqtd3d0e82_pk PRIMARY KEY(date)")
spark.sql(f"COMMENT ON TABLE {TD_TABLE} IS 'Training dataset for PM2.5 regressor'")
print("training dataset rows:", spark.table(TD_TABLE).count())

# COMMAND ----------
# ---- 3. Train + evaluate + register model airqmodel3d0e82 ----
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

pdf = spark.table(TD_TABLE).orderBy("date").toPandas()
X = pdf[FEATURES].astype(float)
y = pdf["pm25"].astype(float)

n_val = 90  # chronological hold-out, same horizon as the forecast set
X_tr, X_val = X.iloc[:-n_val], X.iloc[-n_val:]
y_tr, y_val = y.iloc[:-n_val], y.iloc[-n_val:]

candidates = {
    "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "hist_gbr": HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.05, max_iter=500,
        l2_regularization=1.0, random_state=42),
    "gbr": GradientBoostingRegressor(
        n_estimators=400, max_depth=2, learning_rate=0.05,
        subsample=0.8, random_state=42),
}

results = {}
for name, model in candidates.items():
    model.fit(X_tr, y_tr)
    rmse = float(np.sqrt(mean_squared_error(y_val, model.predict(X_val))))
    results[name] = rmse
    print(name, "val RMSE:", rmse)

baseline_rmse = float(np.sqrt(mean_squared_error(y_val, np.full(len(y_val), y_tr.mean()))))
print("baseline (train-mean) RMSE:", baseline_rmse)

best_name = min(results, key=results.get)
best_model = candidates[best_name]
val_pred = best_model.predict(X_val)
val_rmse = results[best_name]
val_mae = float(mean_absolute_error(y_val, val_pred))
val_r2 = float(r2_score(y_val, val_pred))
print("best:", best_name, val_rmse)

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment("/Users/benedict@hopsworks.ai/mlpab3e3ee5/airq_experiment")

# refit best model on the full history for the final registered model
best_model.fit(X, y)

with mlflow.start_run(run_name="airq_pm25_regressor") as run:
    mlflow.log_params({"model_type": best_name, "features": ",".join(FEATURES),
                       "n_train": len(X_tr), "n_val": n_val})
    mlflow.log_metrics({"val_rmse": val_rmse, "val_mae": val_mae, "val_r2": val_r2,
                        "baseline_rmse": baseline_rmse, "rmse": val_rmse})
    from mlflow.models.signature import infer_signature
    sig = infer_signature(X.head(5), best_model.predict(X.head(5)))
    mlflow.sklearn.log_model(best_model, "model", signature=sig,
                             input_example=X.head(5),
                             registered_model_name=MODEL_NAME)
    print("registered model, run_id:", run.info.run_id)

# COMMAND ----------
# ---- 4. Predict forecast days into airqpred3d0e82 ----
fcst_feat = add_features(fcst).orderBy("date")
fpdf = fcst_feat.toPandas()
preds = best_model.predict(fpdf[FEATURES].astype(float))
out = pd.DataFrame({"date": fpdf["date"], "pm25_pred": preds.astype(float)})
sdf = spark.createDataFrame(out).withColumn("date", F.to_date("date"))

(sdf.write.mode("overwrite").option("overwriteSchema", "true")
    .option("delta.enableChangeDataFeed", "true").saveAsTable(PRED_TABLE))
spark.sql(f"ALTER TABLE {PRED_TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
spark.sql(f"ALTER TABLE {PRED_TABLE} ALTER COLUMN date SET NOT NULL")
spark.sql(f"ALTER TABLE {PRED_TABLE} ADD CONSTRAINT airqpred3d0e82_pk PRIMARY KEY(date)")
spark.sql(f"COMMENT ON TABLE {PRED_TABLE} IS 'PM2.5 predictions for forecast days'")
print("prediction rows:", spark.table(PRED_TABLE).count())
display(spark.table(PRED_TABLE).orderBy("date").limit(5))
print("DONE val_rmse=", val_rmse, "best=", best_name)
