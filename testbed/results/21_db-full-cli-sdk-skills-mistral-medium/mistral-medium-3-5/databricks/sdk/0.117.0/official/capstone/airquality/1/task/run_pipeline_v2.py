#!/usr/bin/env python3
"""
Full FTI pipeline for air quality PM2.5 forecasting.
"""

import os
import time

# Configuration
SCHEMA = os.getenv('MLPAB_DATABRICKS_SCHEMA', 'workspace.mlpab63c012')
PREFIX = os.getenv('MLPAB_DATABRICKS_PREFIX', 'mlpab63c012')
USER = "wolffbe"

FEATURE_GROUP_NAME = "airqb70a96"
TRAINING_DATASET_NAME = "airqtdb70a96"
MODEL_NAME = "airqmodelb70a96"
PREDICTIONS_TABLE_NAME = "airqpredb70a96"

FEATURE_TABLE = f"{SCHEMA}.{FEATURE_GROUP_NAME}"
TRAINING_TABLE = f"{SCHEMA}.{TRAINING_DATASET_NAME}"
PREDICTIONS_TABLE = f"{SCHEMA}.{PREDICTIONS_TABLE_NAME}"
ONLINE_TABLE = f"{SCHEMA}.{PREDICTIONS_TABLE_NAME}_online"
WORKSPACE_PATH = f"/{PREFIX}"

print(f"Schema: {SCHEMA}")
print(f"Feature Group: {FEATURE_TABLE}")
print(f"Training Dataset: {TRAINING_TABLE}")
print(f"Model: {MODEL_NAME}")
print(f"Predictions Table: {PREDICTIONS_TABLE}")
print(f"Online Table: {ONLINE_TABLE}")

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ImportFormat
from databricks.sdk.service import jobs

w = WorkspaceClient()
print(f"Connected to: {w.config.host}")

# Ensure schema exists
try:
    w.schemas.get(SCHEMA)
    print(f"Schema {SCHEMA} exists")
except:
    parts = SCHEMA.split('.')
    w.schemas.create(name=parts[1], catalog_name=parts[0], comment="MLPAB")
    print(f"Created schema {SCHEMA}")

# Upload data files
print("\nUploading data files...")
w.workspace.mkdirs(WORKSPACE_PATH)

with open('data/airquality_history.csv', 'rb') as f:
    w.workspace.upload(path=f"{WORKSPACE_PATH}/airquality_history.csv", content=f, format=ImportFormat.RAW, overwrite=True)
print("Uploaded airquality_history.csv")

with open('data/forecast_days.csv', 'rb') as f:
    w.workspace.upload(path=f"{WORKSPACE_PATH}/forecast_days.csv", content=f, format=ImportFormat.RAW, overwrite=True)
print("Uploaded forecast_days.csv")

# Create and run notebook
print("\nCreating and running pipeline notebook...")

notebook_path = f"{WORKSPACE_PATH}/airq_pipeline"

# Create notebook content - use .format() to insert table names
notebook_content = """# Databricks notebook source
# MAGIC %md ## Air Quality PM2.5 Pipeline

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.window import Window
from pyspark.sql.functions import avg, stddev
import mlflow
import mlflow.spark

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

# Load data
history_df = spark.read.csv("/{PREFIX}/airquality_history.csv", header=True, inferSchema=True)
forecast_df = spark.read.csv("/{PREFIX}/forecast_days.csv", header=True, inferSchema=True)
print(f"History: {{history_df.count()}} rows, Forecast: {{forecast_df.count()}} rows")

# COMMAND ----------

# Feature engineering
feature_history = history_df.withColumn("temp_squared", col("temperature") * col("temperature")) \\
    .withColumn("humidity_squared", col("humidity") * col("humidity")) \\
    .withColumn("wind_speed_squared", col("wind_speed") * col("wind_speed")) \\
    .withColumn("pressure_squared", col("pressure") * col("pressure")) \\
    .withColumn("temp_humidity", col("temperature") * col("humidity")) \\
    .withColumn("temp_wind", col("temperature") * col("wind_speed")) \\
    .withColumn("humidity_wind", col("humidity") * col("wind_speed"))

window_spec = Window.orderBy("date").rowsBetween(-2, 0)
feature_history = feature_history.withColumn("rolling_avg_3d", avg("pm25").over(window_spec)) \\
    .withColumn("rolling_std_3d", stddev("pm25").over(window_spec))

# Save feature group
feature_history.write.mode("overwrite").saveAsTable("{FEATURE_TABLE}")
print(f"Feature group saved: {FEATURE_TABLE}")

# COMMAND ----------

# Create training dataset
training_df = feature_history.select(
    "date", "pm25_lag1", "temperature", "humidity", "wind_speed", "pressure", "precipitation",
    "temp_squared", "humidity_squared", "wind_speed_squared", "pressure_squared",
    "temp_humidity", "temp_wind", "humidity_wind", "rolling_avg_3d", "rolling_std_3d",
    col("pm25").alias("label")
).filter(col("pm25").isNotNull())

training_df.write.mode("overwrite").saveAsTable("{TRAINING_TABLE}")
print(f"Training dataset saved: {TRAINING_TABLE}")

# COMMAND ----------

# Train model
feature_cols = [
    'pm25_lag1', 'temperature', 'humidity', 'wind_speed', 'pressure', 'precipitation',
    'temp_squared', 'humidity_squared', 'wind_speed_squared', 'pressure_squared',
    'temp_humidity', 'temp_wind', 'humidity_wind', 'rolling_avg_3d', 'rolling_std_3d'
]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
train_data, test_data = training_df.randomSplit([0.8, 0.2], seed=42)

rf = RandomForestRegressor(labelCol="label", featuresCol="features", numTrees=200, maxDepth=8, minInstancesPerNode=5, seed=42)
pipeline = Pipeline(stages=[assembler, rf])

mlflow.set_experiment(f"/Users/{USER}/{PREFIX}/airq_experiment")

with mlflow.start_run():
    model = pipeline.fit(train_data)
    predictions = model.transform(test_data)
    evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print(f"Test RMSE: {{rmse}}")
    mlflow.log_metric("rmse", rmse)
    mlflow.spark.log_model(model, "model")
    model_uri = f"runs:/{{mlflow.active_run().info.run_id}}/model"
    mlflow.register_model(model_uri, "{MODEL_NAME}")
    print(f"Model registered: {MODEL_NAME}, RMSE: {{rmse}}")

# COMMAND ----------

# Make predictions
model_path = f"/Users/{USER}/{PREFIX}/airq_experiment"
experiment = mlflow.get_experiment_by_name(model_path)
best_run = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["metrics.rmse ASC"], max_results=1)[0]
run_id = best_run.run_id
pipeline_model = mlflow.spark.load_model(f"runs:/{run_id}/model")

# Feature engineering for forecast
feature_forecast = forecast_df.withColumn("temp_squared", col("temperature") * col("temperature")) \\
    .withColumn("humidity_squared", col("humidity") * col("humidity")) \\
    .withColumn("wind_speed_squared", col("wind_speed") * col("wind_speed")) \\
    .withColumn("pressure_squared", col("pressure") * col("pressure")) \\
    .withColumn("temp_humidity", col("temperature") * col("humidity")) \\
    .withColumn("temp_wind", col("temperature") * col("wind_speed")) \\
    .withColumn("humidity_wind", col("humidity") * col("wind_speed"))

# Get rolling stats
last_stats = training_df.select(avg("label").alias("last_avg_3d"), stddev("label").alias("last_std_3d")).collect()[0]
last_avg_3d = float(last_stats['last_avg_3d']) if last_stats['last_avg_3d'] else 5.0
last_std_3d = float(last_stats['last_std_3d']) if last_stats['last_std_3d'] else 2.0

feature_forecast = feature_forecast.withColumn("rolling_avg_3d", lit(last_avg_3d)) \\
    .withColumn("rolling_std_3d", lit(last_std_3d))

assembler_forecast = VectorAssembler(inputCols=feature_cols, outputCol="features")
forecast_featurized = assembler_forecast.transform(feature_forecast)
predictions = pipeline_model.transform(forecast_featurized)

# Save predictions
predictions.select(col("date"), col("prediction").alias("pm25_pred")) \\
    .write.mode("overwrite").saveAsTable("{PREDICTIONS_TABLE}")
print(f"Predictions saved: {PREDICTIONS_TABLE}, Count: {{predictions.count()}}")

# COMMAND ----------

# Create online table
spark.sql(f"CREATE ONLINE TABLE IF NOT EXISTS {ONLINE_TABLE} SOURCE TABLE {PREDICTIONS_TABLE} PRIMARY KEY (date)")
print(f"Online table created: {ONLINE_TABLE}")

# COMMAND ----------

print("Pipeline complete!")
"""

# Replace placeholders in notebook content
notebook_content = notebook_content.replace("{PREFIX}", PREFIX)
notebook_content = notebook_content.replace("{USER}", USER)
notebook_content = notebook_content.replace("{FEATURE_TABLE}", FEATURE_TABLE)
notebook_content = notebook_content.replace("{TRAINING_TABLE}", TRAINING_TABLE)
notebook_content = notebook_content.replace("{PREDICTIONS_TABLE}", PREDICTIONS_TABLE)
notebook_content = notebook_content.replace("{ONLINE_TABLE}", ONLINE_TABLE)
notebook_content = notebook_content.replace("{MODEL_NAME}", MODEL_NAME)

w.workspace.upload(
    path=notebook_path,
    content=notebook_content.encode('utf-8'),
    overwrite=True,
    format=ImportFormat.JUPYTER
)
print(f"Notebook uploaded: {notebook_path}")

# Submit job
job = w.jobs.submit(
    tasks=[
        jobs.SubmitTask(
            task_key="pipeline",
            notebook_task=jobs.NotebookTask(
                notebook_path=notebook_path,
                warehouse_id="a832b544eb7dc3fe"
            )
        )
    ],
    timeout_seconds=7200
)
print(f"Job submitted: {job.run_id}")

# Wait for completion
for i in range(360):
    run = w.jobs.get_run(job.run_id)
    if run.state.life_cycle_state in ['TERMINATED', 'SKIPPED', 'INTERNAL_ERROR']:
        print(f"Job {run.state.life_cycle_state}: {run.state.result_state}")
        if run.state.result_state == 'SUCCESS':
            print("SUCCESS!")
        else:
            print(f"Failed: {run.state.state_message}")
        break
    time.sleep(10)
    if (i + 1) % 6 == 0:
        print(f"Waiting... ({i+1}/360)")

print(f"\nDone. Feature Group: {FEATURE_GROUP_NAME}, Training: {TRAINING_DATASET_NAME}, Model: {MODEL_NAME}, Predictions: {PREDICTIONS_TABLE_NAME}")
