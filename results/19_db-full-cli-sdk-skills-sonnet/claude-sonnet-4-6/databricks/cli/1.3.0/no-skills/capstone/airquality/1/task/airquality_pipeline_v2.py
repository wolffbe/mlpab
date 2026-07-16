# Databricks notebook source

schema = "workspace.mlpabd7768b"
prefix = "mlpabd7768b"
user = "benedict@logicalclocks.com"
volume_path = "/Volumes/workspace/mlpabd7768b/airqdata"

fg_name = "airqfdfb59"
td_name = "airqtdfdfb59"
model_name = "airqmodelfdfb59"
pred_name = "airqpredfdfb59"

print("Starting pipeline v2...")

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window

# Load raw history data
history_df = spark.read.csv(f"{volume_path}/airquality_history.csv", header=True, inferSchema=True)
history_df = history_df.withColumn("date", F.col("date").cast("date"))
print(f"History rows: {history_df.count()}")

# COMMAND ----------

# Feature engineering with window functions
w = Window.orderBy("date")

history_fe = history_df \
    .withColumn("pm25_lag2", F.lag("pm25", 2).over(w)) \
    .withColumn("pm25_lag3", F.lag("pm25", 3).over(w)) \
    .withColumn("pm25_lag7", F.lag("pm25", 7).over(w)) \
    .withColumn("pm25_rolling3_mean", F.avg("pm25").over(w.rowsBetween(-3, -1))) \
    .withColumn("pm25_rolling7_mean", F.avg("pm25").over(w.rowsBetween(-7, -1))) \
    .withColumn("pm25_rolling14_mean", F.avg("pm25").over(w.rowsBetween(-14, -1))) \
    .withColumn("pm25_rolling7_std", F.stddev("pm25").over(w.rowsBetween(-7, -1))) \
    .withColumn("temp_humidity_ratio", F.col("temperature") / (F.col("humidity") + 1.0)) \
    .withColumn("wind_pressure_ratio", F.col("wind_speed") / (F.col("pressure") + 1.0)) \
    .withColumn("month", F.month("date")) \
    .withColumn("day_of_year", F.dayofyear("date")) \
    .withColumn("day_of_week", F.dayofweek("date")) \
    .withColumn("temp_wind_interaction", F.col("temperature") * F.col("wind_speed")) \
    .withColumn("humidity_precip_interaction", F.col("humidity") * F.col("precipitation"))

# Drop rows with nulls from lag features
history_fe = history_fe.dropna()
print(f"Feature-engineered rows: {history_fe.count()}")

# COMMAND ----------

# Write feature group
spark.sql(f"DROP TABLE IF EXISTS {schema}.{fg_name}")
history_fe.write.format("delta").mode("overwrite").option("overwriteSchema","true").saveAsTable(f"{schema}.{fg_name}")

spark.sql(f"""
  ALTER TABLE {schema}.{fg_name}
  SET TBLPROPERTIES ('feature_group' = 'true', 'primary_key' = 'date', 'delta.enableChangeDataFeed' = 'true')
""")

count = spark.table(f"{schema}.{fg_name}").count()
print(f"Feature group {fg_name}: {count} rows")

# COMMAND ----------

# Assemble training dataset
feature_cols = [
    "date", "pm25_lag1", "pm25_lag2", "pm25_lag3", "pm25_lag7",
    "pm25_rolling3_mean", "pm25_rolling7_mean", "pm25_rolling14_mean", "pm25_rolling7_std",
    "temperature", "humidity", "wind_speed", "pressure", "precipitation",
    "temp_humidity_ratio", "wind_pressure_ratio",
    "month", "day_of_year", "day_of_week",
    "temp_wind_interaction", "humidity_precip_interaction",
    "pm25"
]

td_df = spark.table(f"{schema}.{fg_name}").select(*feature_cols)
spark.sql(f"DROP TABLE IF EXISTS {schema}.{td_name}")
td_df.write.format("delta").mode("overwrite").option("overwriteSchema","true").saveAsTable(f"{schema}.{td_name}")
print(f"Training dataset {td_name}: {td_df.count()} rows")

# COMMAND ----------

import mlflow
import mlflow.spark
import os
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

# Required for serverless/shared clusters
os.environ["MLFLOW_DFS_TMP"] = f"{volume_path}/mlflow_tmp"

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(f"/Users/{user}/{prefix}/airquality_pm25")

feature_input_cols = [
    "pm25_lag1", "pm25_lag2", "pm25_lag3", "pm25_lag7",
    "pm25_rolling3_mean", "pm25_rolling7_mean", "pm25_rolling14_mean", "pm25_rolling7_std",
    "temperature", "humidity", "wind_speed", "pressure", "precipitation",
    "temp_humidity_ratio", "wind_pressure_ratio",
    "month", "day_of_year", "day_of_week",
    "temp_wind_interaction", "humidity_precip_interaction"
]
label_col = "pm25"

# Train/test split by date (80/20 chronological)
all_df = spark.table(f"{schema}.{td_name}").orderBy("date")
total = all_df.count()
split_idx = int(total * 0.8)

rn = Window.orderBy("date")
all_df_rn = all_df.withColumn("rn", F.row_number().over(rn))
train_df = all_df_rn.filter(F.col("rn") <= split_idx).drop("rn")
test_df = all_df_rn.filter(F.col("rn") > split_idx).drop("rn")

print(f"Train: {train_df.count()}, Test: {test_df.count()}")

# COMMAND ----------

# Try GBT with optimized hyperparameters
assembler = VectorAssembler(inputCols=feature_input_cols, outputCol="features", handleInvalid="keep")
gbt = GBTRegressor(
    featuresCol="features",
    labelCol=label_col,
    maxIter=200,
    maxDepth=6,
    stepSize=0.02,
    subsamplingRate=0.8,
    featureSubsetStrategy="sqrt",
    seed=42
)
pipeline_gbt = Pipeline(stages=[assembler, gbt])

with mlflow.start_run(run_name="gbt_pm25_v2") as run:
    model_gbt = pipeline_gbt.fit(train_df)

    preds = model_gbt.transform(test_df)
    rmse = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="rmse").evaluate(preds)
    mae = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="mae").evaluate(preds)
    r2 = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="r2").evaluate(preds)

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    mlflow.log_param("model", "GBT")
    mlflow.log_param("maxIter", 200)
    mlflow.log_param("maxDepth", 6)
    mlflow.log_param("stepSize", 0.02)

    mlflow.spark.log_model(model_gbt, "model", input_example=train_df.limit(5).toPandas()[feature_input_cols])

    run_id_gbt = run.info.run_id
    print(f"GBT RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

# COMMAND ----------

# Try Random Forest
rf = RandomForestRegressor(
    featuresCol="features",
    labelCol=label_col,
    numTrees=300,
    maxDepth=10,
    featureSubsetStrategy="sqrt",
    seed=42
)
pipeline_rf = Pipeline(stages=[assembler, rf])

with mlflow.start_run(run_name="rf_pm25") as run:
    model_rf = pipeline_rf.fit(train_df)

    preds_rf = model_rf.transform(test_df)
    rmse_rf = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="rmse").evaluate(preds_rf)
    mae_rf = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="mae").evaluate(preds_rf)
    r2_rf = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="r2").evaluate(preds_rf)

    mlflow.log_metric("rmse", rmse_rf)
    mlflow.log_metric("mae", mae_rf)
    mlflow.log_metric("r2", r2_rf)
    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("numTrees", 300)
    mlflow.log_param("maxDepth", 10)

    mlflow.spark.log_model(model_rf, "model", input_example=train_df.limit(5).toPandas()[feature_input_cols])

    run_id_rf = run.info.run_id
    print(f"RF RMSE: {rmse_rf:.4f}, MAE: {mae_rf:.4f}, R2: {r2_rf:.4f}")

# COMMAND ----------

# Pick best model
if rmse <= rmse_rf:
    best_model = model_gbt
    best_rmse = rmse
    best_run_id = run_id_gbt
    best_name = "GBT"
else:
    best_model = model_rf
    best_rmse = rmse_rf
    best_run_id = run_id_rf
    best_name = "RandomForest"

print(f"Best model: {best_name} with RMSE={best_rmse:.4f}")

# COMMAND ----------

from mlflow.tracking import MlflowClient

full_model_name = f"{schema}.{model_name}"
model_uri = f"runs:/{best_run_id}/model"

# Register best model
registered = mlflow.register_model(model_uri, full_model_name)
print(f"Registered: {full_model_name} v{registered.version}")

client = MlflowClient()
client.set_registered_model_alias(full_model_name, "champion", registered.version)
client.update_model_version(
    name=full_model_name,
    version=registered.version,
    description=f"{best_name} PM2.5 regressor. RMSE={best_rmse:.4f}"
)

# COMMAND ----------

# Load forecast data and engineer same features
forecast_raw = spark.read.csv(f"{volume_path}/forecast_days.csv", header=True, inferSchema=True)
forecast_raw = forecast_raw.withColumn("date", F.col("date").cast("date"))
print(f"Forecast rows: {forecast_raw.count()}")

# Get actual last values from history for better imputation
# Use the last known pm25 values for rolling features
last_rows = history_df.orderBy(F.col("date").desc()).limit(14).toPandas()
last_pm25_vals = last_rows["pm25"].values.tolist()

# Compute means from actual last historical values
lag2_mean = float(last_rows["pm25"].iloc[1]) if len(last_rows) > 1 else float(last_rows["pm25"].mean())
lag3_mean = float(last_rows["pm25"].iloc[2]) if len(last_rows) > 2 else float(last_rows["pm25"].mean())
lag7_mean = float(last_rows["pm25"].iloc[6]) if len(last_rows) > 6 else float(last_rows["pm25"].mean())
roll3_mean = float(last_rows["pm25"].iloc[:3].mean())
roll7_mean = float(last_rows["pm25"].iloc[:7].mean())
roll14_mean = float(last_rows["pm25"].iloc[:14].mean())
roll7_std = float(last_rows["pm25"].iloc[:7].std()) if len(last_rows) >= 7 else 1.0

print(f"Imputation values: lag2={lag2_mean:.2f}, lag3={lag3_mean:.2f}, lag7={lag7_mean:.2f}, roll3={roll3_mean:.2f}, roll7={roll7_mean:.2f}")

forecast_fe = forecast_raw \
    .withColumn("pm25_lag2", F.lit(lag2_mean)) \
    .withColumn("pm25_lag3", F.lit(lag3_mean)) \
    .withColumn("pm25_lag7", F.lit(lag7_mean)) \
    .withColumn("pm25_rolling3_mean", F.lit(roll3_mean)) \
    .withColumn("pm25_rolling7_mean", F.lit(roll7_mean)) \
    .withColumn("pm25_rolling14_mean", F.lit(roll14_mean)) \
    .withColumn("pm25_rolling7_std", F.lit(roll7_std)) \
    .withColumn("temp_humidity_ratio", F.col("temperature") / (F.col("humidity") + 1.0)) \
    .withColumn("wind_pressure_ratio", F.col("wind_speed") / (F.col("pressure") + 1.0)) \
    .withColumn("month", F.month("date")) \
    .withColumn("day_of_year", F.dayofyear("date")) \
    .withColumn("day_of_week", F.dayofweek("date")) \
    .withColumn("temp_wind_interaction", F.col("temperature") * F.col("wind_speed")) \
    .withColumn("humidity_precip_interaction", F.col("humidity") * F.col("precipitation"))

# COMMAND ----------

# Generate predictions
pred_result = best_model.transform(forecast_fe).select(
    F.col("date"),
    F.col("prediction").alias("pm25_pred")
)

spark.sql(f"DROP TABLE IF EXISTS {schema}.{pred_name}")
pred_result.write.format("delta").mode("overwrite").option("overwriteSchema","true").saveAsTable(f"{schema}.{pred_name}")

spark.sql(f"""
  ALTER TABLE {schema}.{pred_name}
  SET TBLPROPERTIES (
    'delta.enableChangeDataFeed' = 'true',
    'primary_key' = 'date'
  )
""")

print(f"Predictions written: {pred_result.count()} rows")
spark.table(f"{schema}.{pred_name}").show()

# COMMAND ----------

print("=" * 60)
print("PIPELINE V2 COMPLETE")
print("=" * 60)
print(f"Feature group: {schema}.{fg_name}")
print(f"Training dataset: {schema}.{td_name}")
print(f"Model: {schema}.{model_name} (v{registered.version})")
print(f"Predictions: {schema}.{pred_name}")
print(f"Best model: {best_name}")
print(f"Test RMSE: {best_rmse:.4f}")
print(f"Test GBT RMSE: {rmse:.4f}")
print(f"Test RF RMSE: {rmse_rf:.4f}")
