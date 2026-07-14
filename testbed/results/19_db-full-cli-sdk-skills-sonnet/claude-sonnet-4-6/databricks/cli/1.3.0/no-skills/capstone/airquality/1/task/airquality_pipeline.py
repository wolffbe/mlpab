# Databricks notebook source

schema = "workspace.mlpabd7768b"
prefix = "mlpabd7768b"
user = "benedict@logicalclocks.com"
volume_path = "/Volumes/workspace/mlpabd7768b/airqdata"

fg_name = "airqfdfb59"
td_name = "airqtdfdfb59"
model_name = "airqmodelfdfb59"
pred_name = "airqpredfdfb59"

print("Starting pipeline...")
print(f"Schema: {schema}")

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window

# Load raw history data
history_df = spark.read.csv(f"{volume_path}/airquality_history.csv", header=True, inferSchema=True)
history_df = history_df.withColumn("date", F.col("date").cast("date"))
print(f"History rows: {history_df.count()}")
history_df.printSchema()

# COMMAND ----------

# Feature engineering with window functions
w = Window.orderBy("date")

history_fe = history_df \
    .withColumn("pm25_lag2", F.lag("pm25", 2).over(w)) \
    .withColumn("pm25_rolling3_mean", F.avg("pm25").over(w.rowsBetween(-3, -1))) \
    .withColumn("pm25_rolling7_mean", F.avg("pm25").over(w.rowsBetween(-7, -1))) \
    .withColumn("temp_humidity_ratio", F.col("temperature") / (F.col("humidity") + 1.0)) \
    .withColumn("wind_pressure_ratio", F.col("wind_speed") / (F.col("pressure") + 1.0))

# Drop rows with nulls from lag features
history_fe = history_fe.dropna()
print(f"Feature-engineered rows: {history_fe.count()}")

# COMMAND ----------

# Write feature group
spark.sql(f"DROP TABLE IF EXISTS {schema}.{fg_name}")
history_fe.write.format("delta").mode("overwrite").option("overwriteSchema","true").saveAsTable(f"{schema}.{fg_name}")

spark.sql(f"""
  ALTER TABLE {schema}.{fg_name}
  SET TBLPROPERTIES ('feature_group' = 'true', 'primary_key' = 'date')
""")

count = spark.table(f"{schema}.{fg_name}").count()
print(f"Feature group {fg_name}: {count} rows")

# COMMAND ----------

# Assemble training dataset
feature_cols = [
    "date", "pm25_lag1", "pm25_lag2",
    "pm25_rolling3_mean", "pm25_rolling7_mean",
    "temperature", "humidity", "wind_speed", "pressure", "precipitation",
    "temp_humidity_ratio", "wind_pressure_ratio",
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
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

# Required for serverless/shared clusters - use UC volume for temporary Spark ML model files
os.environ["MLFLOW_DFS_TMP"] = f"{volume_path}/mlflow_tmp"

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(f"/Users/{user}/{prefix}/airquality_pm25")

feature_input_cols = [
    "pm25_lag1", "pm25_lag2",
    "pm25_rolling3_mean", "pm25_rolling7_mean",
    "temperature", "humidity", "wind_speed", "pressure", "precipitation",
    "temp_humidity_ratio", "wind_pressure_ratio"
]
label_col = "pm25"

# Train/test split by date
all_df = spark.table(f"{schema}.{td_name}").orderBy("date")
total = all_df.count()
split_idx = int(total * 0.8)

# Use row_number for deterministic split
from pyspark.sql.window import Window as W2
rn = W2.orderBy("date")
all_df_rn = all_df.withColumn("rn", F.row_number().over(rn))
train_df = all_df_rn.filter(F.col("rn") <= split_idx).drop("rn")
test_df = all_df_rn.filter(F.col("rn") > split_idx).drop("rn")

print(f"Train: {train_df.count()}, Test: {test_df.count()}")

# COMMAND ----------

assembler = VectorAssembler(inputCols=feature_input_cols, outputCol="features")
gbt = GBTRegressor(
    featuresCol="features",
    labelCol=label_col,
    maxIter=100,
    maxDepth=5,
    stepSize=0.05,
    subsamplingRate=0.8,
    seed=42
)
pipeline = Pipeline(stages=[assembler, gbt])

with mlflow.start_run(run_name="gbt_pm25") as run:
    model = pipeline.fit(train_df)

    preds = model.transform(test_df)
    rmse = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="rmse").evaluate(preds)
    mae = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="mae").evaluate(preds)
    r2 = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="r2").evaluate(preds)

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    mlflow.log_param("maxIter", 100)
    mlflow.log_param("maxDepth", 5)
    mlflow.log_param("stepSize", 0.05)

    # Log the Spark ML model as an artifact
    mlflow.spark.log_model(model, "model", input_example=train_df.limit(5).toPandas()[feature_input_cols])

    run_id = run.info.run_id
    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}, run_id: {run_id}")

# COMMAND ----------

from mlflow.tracking import MlflowClient

full_model_name = f"{schema}.{model_name}"
model_uri = f"runs:/{run_id}/model"

registered = mlflow.register_model(model_uri, full_model_name)
print(f"Registered: {full_model_name} v{registered.version}")

client = MlflowClient()
client.set_registered_model_alias(full_model_name, "champion", registered.version)
client.update_model_version(
    name=full_model_name,
    version=registered.version,
    description=f"GBT PM2.5 regressor. RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}"
)

# COMMAND ----------

# Load forecast data and engineer same features
forecast_raw = spark.read.csv(f"{volume_path}/forecast_days.csv", header=True, inferSchema=True)
forecast_raw = forecast_raw.withColumn("date", F.col("date").cast("date"))
print(f"Forecast rows: {forecast_raw.count()}")

# Get training means for imputation
stats = spark.table(f"{schema}.{td_name}").agg(
    F.mean("pm25_lag2").alias("m_lag2"),
    F.mean("pm25_rolling3_mean").alias("m_roll3"),
    F.mean("pm25_rolling7_mean").alias("m_roll7")
).collect()[0]

forecast_fe = forecast_raw \
    .withColumn("pm25_lag2", F.lit(float(stats["m_lag2"]))) \
    .withColumn("pm25_rolling3_mean", F.lit(float(stats["m_roll3"]))) \
    .withColumn("pm25_rolling7_mean", F.lit(float(stats["m_roll7"]))) \
    .withColumn("temp_humidity_ratio", F.col("temperature") / (F.col("humidity") + 1.0)) \
    .withColumn("wind_pressure_ratio", F.col("wind_speed") / (F.col("pressure") + 1.0))

# COMMAND ----------

# Generate predictions
pred_result = model.transform(forecast_fe).select(
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

# Create online table for low-latency lookup
import requests
import os

host = spark.conf.get("spark.databricks.workspaceUrl", "dbc-2a4591fe-28e4.cloud.databricks.com")
token = dbutils.secrets.get(scope="mlpab", key="token") if False else ""

# Use the REST API via dbutils notebook context
ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
token_val = ctx.apiToken().get()
host_val = ctx.apiUrl().get()

headers = {"Authorization": f"Bearer {token_val}", "Content-Type": "application/json"}

online_table_name = f"{schema}.{pred_name}_online"
online_table_spec = {
    "name": online_table_name,
    "spec": {
        "source_table_full_name": f"{schema}.{pred_name}",
        "primary_key_columns": ["date"],
        "run_triggered": {}
    }
}

resp = requests.post(
    f"{host_val}/api/2.0/online-tables",
    headers=headers,
    json=online_table_spec
)
print(f"Online table creation status: {resp.status_code}")
print(resp.text[:500])

# COMMAND ----------

print("=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)
print(f"Feature group: {schema}.{fg_name}")
print(f"Training dataset: {schema}.{td_name}")
print(f"Model: {schema}.{model_name}")
print(f"Predictions: {schema}.{pred_name}")
print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Test R2: {r2:.4f}")
