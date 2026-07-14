#!/usr/bin/env python3
"""
Full FTI pipeline for air quality PM2.5 forecasting.
Fixed version with proper notebook format.
"""

import os

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

# Create and run notebook - use proper Databricks notebook format
print("\nCreating and running pipeline notebook...")

notebook_path = f"{WORKSPACE_PATH}/airq_pipeline"

# Create notebook content in proper Databricks notebook JSON format
notebook_content = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Air Quality PM2.5 Pipeline"]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "from pyspark.sql import SparkSession\n",
                "from pyspark.sql.functions import col, lit\n",
                "from pyspark.ml.feature import VectorAssembler\n",
                "from pyspark.ml.regression import RandomForestRegressor\n",
                "from pyspark.ml import Pipeline\n",
                "from pyspark.ml.evaluation import RegressionEvaluator\n",
                "from pyspark.sql.window import Window\n",
                "from pyspark.sql.functions import avg, stddev\n",
                "import mlflow\n",
                "import mlflow.spark"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "spark = SparkSession.builder.getOrCreate()\n",
                f"history_df = spark.read.csv('{WORKSPACE_PATH}/airquality_history.csv', header=True, inferSchema=True)\n",
                f"forecast_df = spark.read.csv('{WORKSPACE_PATH}/forecast_days.csv', header=True, inferSchema=True)\n",
                "print(f'History: {{history_df.count()}} rows, Forecast: {{forecast_df.count()}} rows')"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "# Feature engineering\n",
                "feature_history = history_df.withColumn('temp_squared', col('temperature') * col('temperature')) \\\n",
                "    .withColumn('humidity_squared', col('humidity') * col('humidity')) \\\n",
                "    .withColumn('wind_speed_squared', col('wind_speed') * col('wind_speed')) \\\n",
                "    .withColumn('pressure_squared', col('pressure') * col('pressure')) \\\n",
                "    .withColumn('temp_humidity', col('temperature') * col('humidity')) \\\n",
                "    .withColumn('temp_wind', col('temperature') * col('wind_speed')) \\\n",
                "    .withColumn('humidity_wind', col('humidity') * col('wind_speed'))\n",
                "\n",
                "window_spec = Window.orderBy('date').rowsBetween(-2, 0)\n",
                "feature_history = feature_history.withColumn('rolling_avg_3d', avg('pm25').over(window_spec)) \\\n",
                "    .withColumn('rolling_std_3d', stddev('pm25').over(window_spec))\n",
                "\n",
                f"feature_history.write.mode('overwrite').saveAsTable('{FEATURE_TABLE}')\n",
                f"print(f'Feature group saved: {FEATURE_TABLE}')"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "# Create training dataset\n",
                "training_df = feature_history.select(\n",
                "    'date', 'pm25_lag1', 'temperature', 'humidity', 'wind_speed', 'pressure', 'precipitation',\n",
                "    'temp_squared', 'humidity_squared', 'wind_speed_squared', 'pressure_squared',\n",
                "    'temp_humidity', 'temp_wind', 'humidity_wind', 'rolling_avg_3d', 'rolling_std_3d',\n",
                "    col('pm25').alias('label')\n",
                ").filter(col('pm25').isNotNull())\n",
                f"training_df.write.mode('overwrite').saveAsTable('{TRAINING_TABLE}')\n",
                f"print(f'Training dataset saved: {TRAINING_TABLE}')"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "# Train model\n",
                "feature_cols = [\n",
                "    'pm25_lag1', 'temperature', 'humidity', 'wind_speed', 'pressure', 'precipitation',\n",
                "    'temp_squared', 'humidity_squared', 'wind_speed_squared', 'pressure_squared',\n",
                "    'temp_humidity', 'temp_wind', 'humidity_wind', 'rolling_avg_3d', 'rolling_std_3d'\n",
                "]\n",
                "\n",
                "assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')\n",
                "train_data, test_data = training_df.randomSplit([0.8, 0.2], seed=42)\n",
                "\n",
                "rf = RandomForestRegressor(labelCol='label', featuresCol='features', numTrees=200, maxDepth=8, minInstancesPerNode=5, seed=42)\n",
                "pipeline = Pipeline(stages=[assembler, rf])\n",
                "\n",
                f"mlflow.set_experiment('/Users/{USER}/{PREFIX}/airq_experiment')\n",
                "\n",
                "with mlflow.start_run():\n",
                "    model = pipeline.fit(train_data)\n",
                "    predictions = model.transform(test_data)\n",
                "    evaluator = RegressionEvaluator(labelCol='label', predictionCol='prediction', metricName='rmse')\n",
                "    rmse = evaluator.evaluate(predictions)\n",
                "    print(f'Test RMSE: {{rmse}}')\n",
                "    mlflow.log_metric('rmse', rmse)\n",
                "    mlflow.spark.log_model(model, 'model')\n",
                "    model_uri = f'runs:/{{mlflow.active_run().info.run_id}}/model'\n",
                f"    mlflow.register_model(model_uri, '{MODEL_NAME}')\n",
                "    print(f'Model registered: {MODEL_NAME}, RMSE: {{rmse}}')"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "# Make predictions\n",
                f"model_path = '/Users/{USER}/{PREFIX}/airq_experiment'\n",
                "experiment = mlflow.get_experiment_by_name(model_path)\n",
                "best_run = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=['metrics.rmse ASC'], max_results=1)[0]\n",
                "run_id = best_run.run_id\n",
                "pipeline_model = mlflow.spark.load_model(f'runs:/{run_id}/model')\n",
                "\n",
                "# Feature engineering for forecast\n",
                "feature_forecast = forecast_df.withColumn('temp_squared', col('temperature') * col('temperature')) \\\n",
                "    .withColumn('humidity_squared', col('humidity') * col('humidity')) \\\n",
                "    .withColumn('wind_speed_squared', col('wind_speed') * col('wind_speed')) \\\n",
                "    .withColumn('pressure_squared', col('pressure') * col('pressure')) \\\n",
                "    .withColumn('temp_humidity', col('temperature') * col('humidity')) \\\n",
                "    .withColumn('temp_wind', col('temperature') * col('wind_speed')) \\\n",
                "    .withColumn('humidity_wind', col('humidity') * col('wind_speed'))\n",
                "\n",
                "# Get rolling stats from training data\n",
                "last_stats = training_df.select(avg('label').alias('last_avg_3d'), stddev('label').alias('last_std_3d')).collect()[0]\n",
                "last_avg_3d = float(last_stats['last_avg_3d']) if last_stats['last_avg_3d'] else 5.0\n",
                "last_std_3d = float(last_stats['last_std_3d']) if last_stats['last_std_3d'] else 2.0\n",
                "\n",
                "feature_forecast = feature_forecast.withColumn('rolling_avg_3d', lit(last_avg_3d)) \\\n",
                "    .withColumn('rolling_std_3d', lit(last_std_3d))\n",
                "\n",
                "assembler_forecast = VectorAssembler(inputCols=feature_cols, outputCol='features')\n",
                "forecast_featurized = assembler_forecast.transform(feature_forecast)\n",
                "predictions = pipeline_model.transform(forecast_featurized)\n",
                "\n",
                "# Save predictions\n",
                "predictions.select(col('date'), col('prediction').alias('pm25_pred')) \\\n",
                f"    .write.mode('overwrite').saveAsTable('{PREDICTIONS_TABLE}')\n",
                f"print(f'Predictions saved: {PREDICTIONS_TABLE}, Count: {{predictions.count()}}')"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "# Create online table\n",
                f"spark.sql(f'CREATE ONLINE TABLE IF NOT EXISTS {ONLINE_TABLE} SOURCE TABLE {PREDICTIONS_TABLE} PRIMARY KEY (date)')\n",
                f"print(f'Online table created: {ONLINE_TABLE}')\n",
                "print('Pipeline complete!')"
            ]
        }
    ],
    "metadata": {
        "language": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

import json

# Upload the notebook with proper format
w.workspace.upload(
    path=notebook_path,
    content=json.dumps(notebook_content).encode('utf-8'),
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
print(f"\nDone. Feature Group: {FEATURE_GROUP_NAME}, Training: {TRAINING_DATASET_NAME}, Model: {MODEL_NAME}, Predictions: {PREDICTIONS_TABLE_NAME}")