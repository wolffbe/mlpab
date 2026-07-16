#!/usr/bin/env python3
"""
Simple fraud detection pipeline script.
Creates a notebook that does everything and runs it as a job.
"""
import os
import json
import time
import zipfile
import io
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ImportFormat
from databricks.sdk.service.jobs import Task, NotebookTask

# Configuration
SCHEMA = os.environ.get('MLPAB_DATABRICKS_SCHEMA', 'workspace.mlpabf21a49')
PREFIX = os.environ.get('MLPAB_DATABRICKS_PREFIX', 'mlpabf21a49')

FEATURE_GROUP = 'cctxn0802aa'
TRAINING_DATASET = 'cctd0802aa'
MODEL_NAME = 'ccmodel0802aa'
PREDICTIONS_TABLE = 'ccpred0802aa'

wc = WorkspaceClient()
current_user = wc.current_user.me()
USER_HOME = f'/Users/{current_user.user_name}'
WORKSPACE_PATH = f'{USER_HOME}/{PREFIX}'
WAREHOUSE_ID = 'a832b544eb7dc3fe'

def main():
    print("Starting fraud detection pipeline...")
    print(f"Schema: {SCHEMA}")
    print(f"Workspace path: {WORKSPACE_PATH}")
    
    # Step 1: Upload CSV files to workspace as zip files
    print("\n=== Step 1: Uploading CSV files ===")
    
    # Upload transactions.csv
    with open('data/transactions.csv', 'rb') as f:
        content = f.read()
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        zip_file.writestr('transactions.csv', content)
    
    try:
        wc.workspace.upload(f'{WORKSPACE_PATH}/transactions.zip', zip_buffer.getvalue(), format=ImportFormat.AUTO)
        print("transactions.csv uploaded")
    except Exception as e:
        if "already exists" in str(e):
            print("transactions.csv already exists, skipping")
        else:
            raise
    
    # Upload score_transactions.csv
    with open('data/score_transactions.csv', 'rb') as f:
        content = f.read()
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        zip_file.writestr('score_transactions.csv', content)
    
    try:
        wc.workspace.upload(f'{WORKSPACE_PATH}/score_transactions.zip', zip_buffer.getvalue(), format=ImportFormat.AUTO)
        print("score_transactions.csv uploaded")
    except Exception as e:
        if "already exists" in str(e):
            print("score_transactions.csv already exists, skipping")
        else:
            raise
    
    # Step 2: Create the main notebook
    print("\n=== Step 2: Creating main notebook ===")
    
    # Create notebook content without f-string issues
    notebook_content = f"""# Databricks notebook source
# MAGIC %md ## Credit Card Fraud Detection Pipeline

# COMMAND ----------

# MAGIC %md ### Configuration

# COMMAND ----------

SCHEMA = "{SCHEMA}"
FEATURE_GROUP = "{FEATURE_GROUP}"
TRAINING_DATASET = "{TRAINING_DATASET}"
MODEL_NAME = "{MODEL_NAME}"
PREDICTIONS_TABLE = "{PREDICTIONS_TABLE}"
WORKSPACE_PATH = "{WORKSPACE_PATH}"

print("Schema:", SCHEMA)
print("Workspace path:", WORKSPACE_PATH)

# COMMAND ----------

# MAGIC %md ### Step 1: Load and Extract Data

# COMMAND ----------

import os
import zipfile
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import FloatType
import math

# Extract CSV files from zip archives
zip_files = [
    (WORKSPACE_PATH + "/transactions.zip", WORKSPACE_PATH + "/transactions.csv"),
    (WORKSPACE_PATH + "/score_transactions.zip", WORKSPACE_PATH + "/score_transactions.csv")
]

for zip_path, csv_path in zip_files:
    if os.path.exists(zip_path) and not os.path.exists(csv_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(zip_path))
        print("Extracted", zip_path, "to", csv_path)

# Load data
print("Loading training data...")
train_df = spark.read.csv(WORKSPACE_PATH + "/transactions.csv", header=True, inferSchema=True)
print("Training data:", train_df.count(), "rows,", len(train_df.columns), "columns")

print("Loading scoring data...")
score_df = spark.read.csv(WORKSPACE_PATH + "/score_transactions.csv", header=True, inferSchema=True)
print("Scoring data:", score_df.count(), "rows,", len(score_df.columns), "columns")

# COMMAND ----------

# MAGIC %md ### Step 2: Feature Engineering

# COMMAND ----------

# Parse datetime
train_df = train_df.withColumn("datetime", F.to_timestamp("datetime"))
score_df = score_df.withColumn("datetime", F.to_timestamp("datetime"))

# Time features
for df in [train_df, score_df]:
    df = df.withColumn("hour_of_day", F.hour("datetime"))
    df = df.withColumn("day_of_week", F.dayofweek("datetime"))
    df = df.withColumn("day_of_month", F.dayofmonth("datetime"))
    df = df.withColumn("month", F.month("datetime"))

# Haversine distance UDF
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

haversine_udf = F.udf(haversine, FloatType())

# Card statistics from training data only
card_stats = train_df.groupBy("cc_num").agg(
    F.avg("lat").alias("card_mean_lat"),
    F.avg("long").alias("card_mean_long"),
    F.avg("amount").alias("card_avg_amount"),
    F.stddev("amount").alias("card_std_amount"),
    F.count("*").alias("card_txn_count")
)

# Merchant and category statistics from training data only
merchant_stats = train_df.groupBy("merchant").agg(F.count("*").alias("merchant_txn_count"))
category_stats = train_df.groupBy("category").agg(F.count("*").alias("category_txn_count"))

# Join statistics with training data
train_df = train_df.join(card_stats, "cc_num", "left") \\
    .join(merchant_stats, "merchant", "left") \\
    .join(category_stats, "category", "left")

# Join statistics with scoring data
score_df = score_df.join(card_stats, "cc_num", "left") \\
    .join(merchant_stats, "merchant", "left") \\
    .join(category_stats, "category", "left")

# Window for time-based features
window_spec_card = Window.partitionBy("cc_num").orderBy("datetime")

# Time since last transaction
for df in [train_df, score_df]:
    df = df.withColumn("time_since_last_txn", 
        F.coalesce(F.unix_timestamp("datetime") - F.lag(F.unix_timestamp("datetime")).over(window_spec_card), F.lit(0)))

# Transaction count in recent history (approximate)
for df in [train_df, score_df]:
    df = df.withColumn("txn_count_recent", 
        F.count("*").over(window_spec_card.rowsBetween(-10, 0)) - 1)

# Geo distance from card mean location
for df in [train_df, score_df]:
    df = df.withColumn("geo_distance", 
        haversine_udf(F.col("lat"), F.col("long"), F.col("card_mean_lat"), F.col("card_mean_long")))

# Amount features
for df in [train_df, score_df]:
    df = df.withColumn("amount_rel_to_avg", 
        F.col("amount") / F.coalesce(F.col("card_avg_amount"), F.lit(1.0)))
    df = df.withColumn("amount_zscore", 
        (F.col("amount") - F.col("card_avg_amount")) / F.coalesce(F.col("card_std_amount"), F.lit(1.0)))

print("Feature engineering complete!")

# COMMAND ----------

# MAGIC %md ### Step 3: Create Feature Group

# COMMAND ----------

feature_columns = [
    'transaction_id', 'cc_num', 'datetime', 'amount', 'merchant', 'category', 'lat', 'long',
    'hour_of_day', 'day_of_week', 'day_of_month', 'month',
    'time_since_last_txn', 'txn_count_recent',
    'card_mean_lat', 'card_mean_long', 'card_avg_amount', 'card_std_amount', 'card_txn_count',
    'merchant_txn_count', 'category_txn_count',
    'geo_distance', 'amount_rel_to_avg', 'amount_zscore'
]

# Create feature group table
fg_table_name = SCHEMA + "." + FEATURE_GROUP
print("Creating feature group:", fg_table_name)
train_df.select(*feature_columns).write.mode("overwrite").saveAsTable(fg_table_name)
print("Feature group", FEATURE_GROUP, "created!")

# COMMAND ----------

# MAGIC %md ### Step 4: Create Training Dataset

# COMMAND ----------

td_table_name = SCHEMA + "." + TRAINING_DATASET
print("Creating training dataset:", td_table_name)
train_df.select(*feature_columns, 'is_fraud').write.mode("overwrite").saveAsTable(td_table_name)
print("Training dataset", TRAINING_DATASET, "created!")

# Check fraud rate
fraud_rate = train_df.filter("is_fraud = 1").count() / train_df.count()
print("Fraud rate:", fraud_rate)

# COMMAND ----------

# MAGIC %md ### Step 5: Train Model

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import mlflow
import mlflow.spark
from mlflow.models.signature import infer_signature

print("Preparing data for ML...")
# Use the training dataset table
ml_df = spark.table(td_table_name).na.drop()
print("ML data:", ml_df.count(), "rows")

# Categorical and numeric columns
categorical_cols = ['merchant', 'category']
numeric_cols = [c for c in feature_columns if c not in categorical_cols + ['transaction_id', 'cc_num', 'datetime']]

print("Categorical cols:", categorical_cols)
print("Numeric cols:", numeric_cols)

# Index categorical columns
indexers = [StringIndexer(inputCol=col, outputCol=col+"_idx", handleInvalid="keep").fit(ml_df) for col in categorical_cols]

# One-hot encode
encoder = OneHotEncoder(inputCols=[col+"_idx" for col in categorical_cols], 
                        outputCols=[col+"_encoded" for col in categorical_cols],
                        handleInvalid="keep")

# Assemble feature vector
feature_cols = numeric_cols + [col+"_encoded" for col in categorical_cols]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# Scale features
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

# Classifier - use more trees and depth for better performance
rf = RandomForestClassifier(featuresCol="scaled_features", labelCol="is_fraud", 
                            numTrees=300, maxDepth=8, seed=42, subsamplingRate=0.8,
                            maxBins=32, minInstancesPerNode=5)

# Pipeline
pipeline = Pipeline(stages=indexers + [encoder, assembler, scaler, rf])

# Split data
(train_data, val_data) = ml_df.randomSplit([0.8, 0.2], seed=42)
print("Training:", train_data.count(), "Validation:", val_data.count())

# Train model
print("Training model...")
model = pipeline.fit(train_data)

# Predict on validation
predictions = model.transform(val_data)

# Evaluate
evaluator = BinaryClassificationEvaluator(labelCol="is_fraud", metricName="areaUnderROC")
roc_auc = evaluator.evaluate(predictions)
print("Validation ROC AUC:", roc_auc)

# Log model with MLflow
print("Logging model to MLflow...")
with mlflow.start_run() as run:
    mlflow.log_param("num_trees", 300)
    mlflow.log_param("max_depth", 8)
    mlflow.log_param("subsampling_rate", 0.8)
    mlflow.log_param("max_bins", 32)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.spark.log_model(model, MODEL_NAME)
    signature = infer_signature(val_data.select("features"), predictions.select("prediction"))
    mlflow.set_signature(signature)
    run_id = run.info.run_id
    print("MLflow run ID:", run_id)

# COMMAND ----------

# MAGIC %md ### Step 6: Register Model

# COMMAND ----------

print("Registering model in Unity Catalog...")
model_uri = "runs:/" + run_id + "/model"
mlflow.set_registry_uri("databricks-uc")
result = mlflow.register_model(
    model_uri=model_uri,
    name=SCHEMA + "." + MODEL_NAME
)
print("Model registered:", result.name, "version:", result.version)

# COMMAND ----------

# MAGIC %md ### Step 7: Score Transactions

# COMMAND ----------

print("Scoring transactions...")
# Apply same feature engineering to score data
score_ml_df = score_df.na.drop()
print("Scoring data:", score_ml_df.count(), "rows")

# Apply the same transformations and predict
score_predictions = model.transform(score_ml_df)

# Extract predictions
predictions_df = score_predictions.select(
    "transaction_id",
    F.col("probability")[1].alias("fraud_probability")
)

print("Predictions:", predictions_df.count(), "rows")
predictions_df.limit(10).show()

# COMMAND ----------

# MAGIC %md ### Step 8: Create Predictions Table

# COMMAND ----------

predictions_table_name = SCHEMA + "." + PREDICTIONS_TABLE
print("Creating predictions table:", predictions_table_name)
predictions_df.write.mode("overwrite").saveAsTable(predictions_table_name)
print("Predictions table", PREDICTIONS_TABLE, "created!")

# Verify predictions
from pyspark.sql.functions import min as spark_min, max as spark_max, avg
predictions_df.agg(spark_min("fraud_probability"), spark_max("fraud_probability"), avg("fraud_probability")).show()

# COMMAND ----------

# MAGIC %md ### Step 9: Create Online Table for Low-Latency Lookup

# COMMAND ----------

print("Creating online table for low-latency lookup...")
try:
    # Create a feature table that can be used for online lookup
    spark.sql("CREATE OR REPLACE TABLE " + SCHEMA + "." + PREDICTIONS_TABLE + "_online AS SELECT transaction_id, fraud_probability FROM " + SCHEMA + "." + PREDICTIONS_TABLE)
    print("Online lookup table created!")
except Exception as e:
    print("Online table creation failed:", e)

# COMMAND ----------

print("\\n=== Pipeline Complete ===")
print("Feature Group:", SCHEMA + "." + FEATURE_GROUP)
print("Training Dataset:", SCHEMA + "." + TRAINING_DATASET)
print("Model:", SCHEMA + "." + MODEL_NAME)
print("Predictions Table:", SCHEMA + "." + PREDICTIONS_TABLE)
print("Validation ROC AUC:", roc_auc)
"""
    
    notebook_path = f"{WORKSPACE_PATH}/fraud_pipeline"
    
    # Create notebook using workspace API
    notebook_json = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# Databricks notebook source\n", "# MAGIC %md ## Credit Card Fraud Detection Pipeline\n"]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": notebook_content
            }
        ],
        "metadata": {
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    try:
        wc.workspace.upload(notebook_path, json.dumps(notebook_json).encode('utf-8'), format=ImportFormat.JUPYTER)
    except Exception as e:
        if "already exists" in str(e):
            print("Notebook already exists, skipping")
        else:
            raise
    print(f"Notebook created at: {notebook_path}")
    
    # Step 3: Create and run job
    print("\n=== Step 3: Creating and running job ===")
    
    job_name = f"{PREFIX}_fraud_pipeline"
    
    # Use serverless warehouse
    task = Task(
        task_key="run_fraud_pipeline",
        notebook_task=NotebookTask(
            notebook_path=notebook_path,
            warehouse_id=WAREHOUSE_ID
        )
    )
    
    job = wc.jobs.create(
        name=job_name,
        tasks=[task]
    )
    
    print(f"Job created: {job.job_id}")
    
    # Run the job
    run = wc.jobs.run_now(job.job_id)
    print(f"Job run started: {run.run_id}")
    
    # Wait for completion
    print("Waiting for job to complete...")
    while True:
        run_info = wc.jobs.get_run(run.run_id)
        state = run_info.state.life_cycle_state
        result_state = run_info.state.result_state
        
        if state in ["TERMINATED", "SKIPPED"]:
            print(f"Job completed with state: {state}, result: {result_state}")
            break
        elif state == "INTERNAL_ERROR":
            print(f"Job failed with internal error: {result_state}")
            break
        
        time.sleep(15)
    
    # Get job output
    try:
        output = wc.jobs.get_run_output(run.run_id)
        print(f"Job output: {output}")
    except Exception as e:
        print(f"Could not get job output: {e}")
    
    print("\n=== Pipeline Complete ===")
    print(f"Feature Group: {SCHEMA}.{FEATURE_GROUP}")
    print(f"Training Dataset: {SCHEMA}.{TRAINING_DATASET}")
    print(f"Model: {SCHEMA}.{MODEL_NAME}")
    print(f"Predictions Table: {SCHEMA}.{PREDICTIONS_TABLE}")

if __name__ == "__main__":
    main()
