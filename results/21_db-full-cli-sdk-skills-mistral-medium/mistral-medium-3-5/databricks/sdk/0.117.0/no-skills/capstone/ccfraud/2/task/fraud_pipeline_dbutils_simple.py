#!/usr/bin/env python3
"""
Create a fraud detection notebook using dbutils.fs.
"""
import os
import json
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
    print("Creating fraud detection notebook with dbutils...")
    
    # Create notebook content
    notebook_content = f"""# Databricks notebook source
# MAGIC %md ## Credit Card Fraud Detection Pipeline

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

# Use dbutils to copy files to DBFS
print("Copying files to DBFS...")
dbutils.fs.cp(WORKSPACE_PATH + "/transactions.csv", "dbfs:/tmp/transactions.csv")
dbutils.fs.cp(WORKSPACE_PATH + "/score_transactions.csv", "dbfs:/tmp/score_transactions.csv")

# Load data from DBFS
print("Loading training data...")
train_df = spark.read.csv("dbfs:/tmp/transactions.csv", header=True, inferSchema=True)
print("Training data:", train_df.count(), "rows")

print("Loading scoring data...")
score_df = spark.read.csv("dbfs:/tmp/score_transactions.csv", header=True, inferSchema=True)
print("Scoring data:", score_df.count(), "rows")

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import FloatType
import math

# Parse datetime
train_df = train_df.withColumn("datetime", F.to_timestamp("datetime"))
score_df = score_df.withColumn("datetime", F.to_timestamp("datetime"))

# Time features
train_df = train_df.withColumn("hour_of_day", F.hour("datetime"))
train_df = train_df.withColumn("day_of_week", F.dayofweek("datetime"))
train_df = train_df.withColumn("day_of_month", F.dayofmonth("datetime"))
train_df = train_df.withColumn("month", F.month("datetime"))

score_df = score_df.withColumn("hour_of_day", F.hour("datetime"))
score_df = score_df.withColumn("day_of_week", F.dayofweek("datetime"))
score_df = score_df.withColumn("day_of_month", F.dayofmonth("datetime"))
score_df = score_df.withColumn("month", F.month("datetime"))

# Haversine distance UDF
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
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
train_df = train_df.withColumn("time_since_last_txn", 
    F.coalesce(F.unix_timestamp("datetime") - F.lag(F.unix_timestamp("datetime")).over(window_spec_card), F.lit(0)))
score_df = score_df.withColumn("time_since_last_txn", 
    F.coalesce(F.unix_timestamp("datetime") - F.lag(F.unix_timestamp("datetime")).over(window_spec_card), F.lit(0)))

# Transaction count in recent history
train_df = train_df.withColumn("txn_count_recent", 
    F.count("*").over(window_spec_card.rowsBetween(-10, 0)) - 1)
score_df = score_df.withColumn("txn_count_recent", 
    F.count("*").over(window_spec_card.rowsBetween(-10, 0)) - 1)

# Geo distance from card mean location
train_df = train_df.withColumn("geo_distance", 
    haversine_udf(F.col("lat"), F.col("long"), F.col("card_mean_lat"), F.col("card_mean_long")))
score_df = score_df.withColumn("geo_distance", 
    haversine_udf(F.col("lat"), F.col("long"), F.col("card_mean_lat"), F.col("card_mean_long")))

# Amount features
train_df = train_df.withColumn("amount_rel_to_avg", 
    F.col("amount") / F.coalesce(F.col("card_avg_amount"), F.lit(1.0)))
train_df = train_df.withColumn("amount_zscore", 
    (F.col("amount") - F.col("card_avg_amount")) / F.coalesce(F.col("card_std_amount"), F.lit(1.0)))

score_df = score_df.withColumn("amount_rel_to_avg", 
    F.col("amount") / F.coalesce(F.col("card_avg_amount"), F.lit(1.0)))
score_df = score_df.withColumn("amount_zscore", 
    (F.col("amount") - F.col("card_avg_amount")) / F.coalesce(F.col("card_std_amount"), F.lit(1.0)))

print("Feature engineering complete!")

# COMMAND ----------

# Create feature group
feature_columns = [
    'transaction_id', 'cc_num', 'datetime', 'amount', 'merchant', 'category', 'lat', 'long',
    'hour_of_day', 'day_of_week', 'day_of_month', 'month',
    'time_since_last_txn', 'txn_count_recent',
    'card_mean_lat', 'card_mean_long', 'card_avg_amount', 'card_std_amount', 'card_txn_count',
    'merchant_txn_count', 'category_txn_count',
    'geo_distance', 'amount_rel_to_avg', 'amount_zscore'
]

fg_table_name = SCHEMA + "." + FEATURE_GROUP
print("Creating feature group:", fg_table_name)
train_df.select(*feature_columns).write.mode("overwrite").saveAsTable(fg_table_name)
print("Feature group created!")

# Create training dataset
td_table_name = SCHEMA + "." + TRAINING_DATASET
print("Creating training dataset:", td_table_name)
train_df.select(*feature_columns, 'is_fraud').write.mode("overwrite").saveAsTable(td_table_name)
print("Training dataset created!")

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import mlflow
import mlflow.spark
from mlflow.models.signature import infer_signature

# Prepare data for ML
ml_df = spark.table(td_table_name).na.drop()
print("ML data:", ml_df.count(), "rows")

# Categorical and numeric columns
categorical_cols = ['merchant', 'category']
numeric_cols = [c for c in feature_columns if c not in categorical_cols + ['transaction_id', 'cc_num', 'datetime']]

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

# Classifier
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
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.spark.log_model(model, MODEL_NAME)
    signature = infer_signature(val_data.select("features"), predictions.select("prediction"))
    mlflow.set_signature(signature)
    run_id = run.info.run_id
    print("MLflow run ID:", run_id)

# Register model
print("Registering model...")
model_uri = "runs:/" + run_id + "/model"
mlflow.set_registry_uri("databricks-uc")
result = mlflow.register_model(
    model_uri=model_uri,
    name=SCHEMA + "." + MODEL_NAME
)
print("Model registered:", result.name, "version:", result.version)

# COMMAND ----------

# Score transactions
print("Scoring transactions...")
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

# Create predictions table
predictions_table_name = SCHEMA + "." + PREDICTIONS_TABLE
print("Creating predictions table:", predictions_table_name)
predictions_df.write.mode("overwrite").saveAsTable(predictions_table_name)
print("Predictions table created!")

# Verify predictions
from pyspark.sql.functions import min as spark_min, max as spark_max, avg
predictions_df.agg(spark_min("fraud_probability"), spark_max("fraud_probability"), avg("fraud_probability")).show()

# COMMAND ----------

# Create online table for low-latency lookup
print("Creating online table...")
try:
    spark.sql("CREATE OR REPLACE TABLE " + SCHEMA + "." + PREDICTIONS_TABLE + "_online AS SELECT transaction_id, fraud_probability FROM " + SCHEMA + "." + PREDICTIONS_TABLE)
    print("Online lookup table created!")
except Exception as e:
    print("Online table creation failed:", e)

# COMMAND ----------

print("Pipeline Complete!")
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
            print("Notebook already exists, updating...")
        else:
            raise
    
    print(f"Notebook created at: {notebook_path}")
    
    # Create and run job
    print("Creating and running job...")
    
    job_name = f"{PREFIX}_fraud_pipeline_dbutils"
    
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
    
    print("Job is running on the platform. Check Databricks workspace for results.")
    print(f"Notebook: {notebook_path}")
    print(f"Job ID: {job.job_id}")
    print(f"Run ID: {run.run_id}")

if __name__ == "__main__":
    main()
