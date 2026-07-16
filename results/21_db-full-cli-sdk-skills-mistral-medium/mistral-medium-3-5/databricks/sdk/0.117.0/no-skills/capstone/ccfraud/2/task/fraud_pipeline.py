#!/usr/bin/env python3
"""
Fraud detection pipeline for Databricks platform.
This script creates and executes the entire FTI pipeline.
"""
import os
import json
from databricks.sdk import WorkspaceClient

# Environment variables
SCHEMA = os.environ.get('MLPAB_DATABRICKS_SCHEMA', 'workspace.mlpabf21a49')
PREFIX = os.environ.get('MLPABRICKS_PREFIX', 'mlpabf21a49')

# Object names
FEATURE_GROUP = 'cctxn0802aa'
TRAINING_DATASET = 'cctd0802aa'
MODEL_NAME = 'ccmodel0802aa'
PREDICTIONS_TABLE = 'ccpred0802aa'

# User home
USER_HOME = '/Users/' + WorkspaceClient().current_user.me().user_name
WORKSPACE_PATH = f'{USER_HOME}/{PREFIX}'

def main():
    wc = WorkspaceClient()
    
    # Step 1: Upload data files to DBFS
    print("Uploading data files to DBFS...")
    wc.dbfs.upload('dbfs:/' + WORKSPACE_PATH + '/data/transactions.csv', 
                   'data/transactions.csv', overwrite=True)
    wc.dbfs.upload('dbfs:/' + WORKSPACE_PATH + '/data/score_transactions.csv', 
                   'data/score_transactions.csv', overwrite=True)
    
    # Step 2: Create the notebook
    notebook_content = f'''# MAGIC %md
# MAGIC ## Credit Card Fraud Detection Pipeline
# MAGIC 
# MAGIC This notebook performs:
# MAGIC 1. Feature engineering for fraud detection
# MAGIC 2. Training dataset assembly
# MAGIC 3. Model training and registration
# MAGIC 4. Scoring of new transactions

# COMMAND ----------

# Import libraries
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import mlflow
import mlflow.spark
from mlflow.models.signature import infer_signature
import pandas as pd
import numpy as np
from datetime import datetime

# COMMAND ----------

# Configuration
SCHEMA = "{SCHEMA}"
FEATURE_GROUP = "{FEATURE_GROUP}"
TRAINING_DATASET = "{TRAINING_DATASET}"
MODEL_NAME = "{MODEL_NAME}"
PREDICTIONS_TABLE = "{PREDICTIONS_TABLE}"
WORKSPACE_PATH = "{WORKSPACE_PATH}"

print(f"Schema: {SCHEMA}")
print(f"Workspace path: {WORKSPACE_PATH}")

# COMMAND ----------

# Step 1: Load the data
print("Loading training data...")
train_df = spark.read.csv(f"{WORKSPACE_PATH}/data/transactions.csv", header=True, inferSchema=True)
print(f"Training data shape: {train_df.count()} rows, {len(train_df.columns)} columns")

print("Loading scoring data...")
score_df = spark.read.csv(f"{WORKSPACE_PATH}/data/score_transactions.csv", header=True, inferSchema=True)
print(f"Scoring data shape: {score_df.count()} rows, {len(score_df.columns)} columns")

# COMMAND ----------

# Step 2: Feature Engineering
print("\\n=== Feature Engineering ===")

# Parse datetime
train_df = train_df.withColumn("datetime", F.to_timestamp("datetime"))
score_df = score_df.withColumn("datetime", F.to_timestamp("datetime"))

# Extract time features
time_features = [
    F.hour("datetime").alias("hour_of_day"),
    F.dayofweek("datetime").alias("day_of_week"),
    F.dayofmonth("datetime").alias("day_of_month"),
    F.month("datetime").alias("month"),
]

for feat in time_features:
    train_df = train_df.withColumn(feat.name, feat)
    score_df = score_df.withColumn(feat.name, feat)

# Card-level features: transaction velocity
window_spec_card = Window.partitionBy("cc_num").orderBy("datetime")

# Time since last transaction for each card
train_df = train_df.withColumn("time_since_last_txn", 
    F.coalesce(F.unix_timestamp("datetime") - F.lag(F.unix_timestamp("datetime")).over(window_spec_card), F.lit(0)))
score_df = score_df.withColumn("time_since_last_txn", 
    F.coalesce(F.unix_timestamp("datetime") - F.lag(F.unix_timestamp("datetime")).over(window_spec_card), F.lit(0)))

# Transaction count per card in last hour
train_df = train_df.withColumn("txn_count_last_hour", 
    F.count("*").over(window_spec_card.rowsBetween(-10, 0)) - 1)  # Approximate
score_df = score_df.withColumn("txn_count_last_hour", 
    F.count("*").over(window_spec_card.rowsBetween(-10, 0)) - 1)

# Amount features
# Rolling average amount per card
train_df = train_df.withColumn("avg_amount_card", 
    F.avg("amount").over(window_spec_card.rowsBetween(Window.unboundedPreceding, 0)))
score_df = score_df.withColumn("avg_amount_card", 
    F.avg("amount").over(window_spec_card.rowsBetween(Window.unboundedPreceding, 0)))

# Amount relative to card average
train_df = train_df.withColumn("amount_rel_to_avg", 
    F.col("amount") / F.coalesce(F.col("avg_amount_card"), F.lit(1.0)))
score_df = score_df.withColumn("amount_rel_to_avg", 
    F.col("amount") / F.coalesce(F.col("avg_amount_card"), F.lit(1.0)))

# Geo features: distance from card's usual location
# Calculate card's mean location
card_loc_window = Window.partitionBy("cc_num")
card_mean_lat = F.avg("lat").over(card_loc_window).alias("card_mean_lat")
card_mean_long = F.avg("long").over(card_loc_window).alias("card_mean_long")

train_df = train_df.withColumn("card_mean_lat", card_mean_lat) \
    .withColumn("card_mean_long", card_mean_long)
score_df = score_df.withColumn("card_mean_lat", card_mean_lat) \
    .withColumn("card_mean_long", card_mean_long)

# Haversine distance function
def haversine(lat1, lon1, lat2, lon2):
    from math import radians, sin, cos, sqrt, atan2
    R = 6371  # Earth radius in km
    dLat = radians(lat2 - lat1)
    dLon = radians(lon2 - lon1)
    a = sin(dLat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dLon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

haversine_udf = F.udf(haversine, FloatType())

train_df = train_df.withColumn("geo_distance", 
    haversine_udf(F.col("lat"), F.col("long"), F.col("card_mean_lat"), F.col("card_mean_long")))
score_df = score_df.withColumn("geo_distance", 
    haversine_udf(F.col("lat"), F.col("long"), F.col("card_mean_lat"), F.col("card_mean_long")))

# Merchant and category frequency features
# Count transactions per merchant (potential fraud indicator)
merchant_count_window = Window.partitionBy("merchant")
train_df = train_df.withColumn("merchant_txn_count", 
    F.count("*").over(merchant_count_window))
score_df = score_df.withColumn("merchant_txn_count", 
    F.count("*").over(merchant_count_window))

# Count transactions per category
category_count_window = Window.partitionBy("category")
train_df = train_df.withColumn("category_txn_count", 
    F.count("*").over(category_count_window))
score_df = score_df.withColumn("category_txn_count", 
    F.count("*").over(category_count_window))

# Amount standard deviation per card
card_std_window = Window.partitionBy("cc_num")
train_df = train_df.withColumn("card_amount_std", 
    F.stddev("amount").over(card_std_window))
score_df = score_df.withColumn("card_amount_std", 
    F.stddev("amount").over(card_std_window))

# Z-score of amount relative to card
train_df = train_df.withColumn("amount_zscore", 
    (F.col("amount") - F.col("avg_amount_card")) / F.coalesce(F.col("card_amount_std"), F.lit(1.0)))
score_df = score_df.withColumn("amount_zscore", 
    (F.col("amount") - F.col("avg_amount_card")) / F.coalesce(F.col("card_amount_std"), F.lit(1.0)))

print("Feature engineering complete!")
print(f"Training features: {[c for c in train_df.columns if c not in ['transaction_id', 'cc_num', 'datetime', 'is_fraud']]}")

# COMMAND ----------

# Step 3: Create Feature Group
print("\\n=== Creating Feature Group ===")

# Select features for the feature group
feature_columns = [
    'transaction_id', 'cc_num', 'datetime', 'amount', 'merchant', 'category', 'lat', 'long',
    'hour_of_day', 'day_of_week', 'day_of_month', 'month',
    'time_since_last_txn', 'txn_count_last_hour', 'avg_amount_card', 'amount_rel_to_avg',
    'geo_distance', 'merchant_txn_count', 'category_txn_count', 'card_amount_std', 'amount_zscore'
]

# Create feature group table
fg_table_name = f"{SCHEMA}.{FEATURE_GROUP}"
print(f"Creating feature group table: {fg_table_name}")

# Write the feature group (we'll use the training data as the source)
feature_df = train_df.select(*feature_columns)
feature_df.write.mode("overwrite").saveAsTable(fg_table_name)

print(f"Feature group {FEATURE_GROUP} created successfully!")

# COMMAND ----------

# Step 4: Create Training Dataset
print("\\n=== Creating Training Dataset ===")

td_table_name = f"{SCHEMA}.{TRAINING_DATASET}"
print(f"Creating training dataset table: {td_table_name}")

# Add label to features
training_data = train_df.select(*feature_columns, 'is_fraud')
training_data.write.mode("overwrite").saveAsTable(td_table_name)

print(f"Training dataset {TRAINING_DATASET} created successfully!")
print(f"Fraud rate: {(training_data.filter('is_fraud = 1').count() / training_data.count()):.4f}")

# COMMAND ----------

# Step 5: Train Model
print("\\n=== Training Model ===")

# Prepare data for ML
# Drop rows with null values in features
ml_df = training_data.na.drop(subset=feature_columns)
print(f"Training samples after dropping nulls: {ml_df.count()}")

# Encode categorical features
categorical_cols = ['merchant', 'category']
numeric_cols = [c for c in feature_columns if c not in categorical_cols + ['transaction_id', 'cc_num', 'datetime']]

# Index categorical columns
indexers = [StringIndexer(inputCol=col, outputCol=col+"_idx").fit(ml_df) for col in categorical_cols]

# One-hot encode
encoder = OneHotEncoder(inputCols=[col+"_idx" for col in categorical_cols], 
                        outputCols=[col+"_encoded" for col in categorical_cols])

# Assemble feature vector
feature_cols = numeric_cols + [col+"_encoded" for col in categorical_cols]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# Scale features
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

# Classifier
rf = RandomForestClassifier(featuresCol="scaled_features", labelCol="is_fraud", 
                            numTrees=100, maxDepth=5, seed=42)

# Pipeline
pipeline = Pipeline(stages=indexers + [encoder, assembler, scaler, rf])

# Split data
(train_data, val_data) = ml_df.randomSplit([0.8, 0.2], seed=42)

print(f"Training samples: {train_data.count()}")
print(f"Validation samples: {val_data.count()}")

# Train model
print("Training model...")
model = pipeline.fit(train_data)

# Predict on validation
predictions = model.transform(val_data)

# Evaluate
evaluator = BinaryClassificationEvaluator(labelCol="is_fraud", metricName="areaUnderROC")
roc_auc = evaluator.evaluate(predictions)
print(f"Validation ROC AUC: {roc_auc:.4f}")

# Log model with MLflow
print("\\nLogging model to MLflow...")
with mlflow.start_run() as run:
    # Log parameters
    mlflow.log_param("num_trees", 100)
    mlflow.log_param("max_depth", 5)
    
    # Log metrics
    mlflow.log_metric("roc_auc", roc_auc)
    
    # Log model
    mlflow.spark.log_model(model, MODEL_NAME)
    
    # Set model signature
    signature = infer_signature(val_data.select("features"), predictions.select("prediction"))
    mlflow.set_signature(signature)
    
    # Get run ID
    run_id = run.info.run_id
    print(f"MLflow run ID: {run_id}")

# COMMAND ----------

# Step 6: Register Model in Unity Catalog
print("\\n=== Registering Model ===")

# Get the best model from MLflow
model_uri = f"runs:/{run_id}/model"

# Register model in Unity Catalog
mlflow.set_registry_uri("databricks-uc")
result = mlflow.register_model(
    model_uri=model_uri,
    name=f"{SCHEMA}.{MODEL_NAME}"
)

print(f"Model registered: {result.name}")
print(f"Model version: {result.version}")

# COMMAND ----------

# Step 7: Score New Transactions
print("\\n=== Scoring New Transactions ===")

# Prepare scoring data
score_ml_df = score_df.na.drop(subset=feature_columns)
print(f"Scoring samples after dropping nulls: {score_ml_df.count()}")

# Apply the same transformations and predict
score_predictions = model.transform(score_ml_df)

# Extract predictions
predictions_df = score_predictions.select(
    "transaction_id",
    F.col("probability")[1].alias("fraud_probability")  # Probability of class 1 (fraud)
)

print(f"Predictions shape: {predictions_df.count()} rows")
print("Sample predictions:")
predictions_df.limit(10).show()

# COMMAND ----------

# Step 8: Create Predictions Table
print("\\n=== Creating Predictions Table ===")

predictions_table_name = f"{SCHEMA}.{PREDICTIONS_TABLE}"
print(f"Creating predictions table: {predictions_table_name}")

predictions_df.write.mode("overwrite").saveAsTable(predictions_table_name)

print(f"Predictions table {PREDICTIONS_TABLE} created successfully!")

# COMMAND ----------

# Step 9: Create Online Table for Low-Latency Lookup
print("\\n=== Creating Online Table ===")

# Create online table from predictions
online_table_name = f"{PREDICTIONS_TABLE}_online"

# First, ensure the predictions table exists as a Delta table
predictions_df.write.mode("overwrite").format("delta").saveAsTable(predictions_table_name)

# Create online table
try:
    spark.sql(f"""
    CREATE ONLINE TABLE IF NOT EXISTS {SCHEMA}.{online_table_name} 
    AS SELECT * FROM {SCHEMA}.{PREDICTIONS_TABLE}
    """)
    print(f"Online table {online_table_name} created successfully!")
except Exception as e:
    print(f"Could not create online table (may not be supported): {e}")
    # Try alternative approach using feature store
    try:
        spark.sql(f"""
        CREATE FEATURE TABLE IF NOT EXISTS {SCHEMA}.{online_table_name} 
        AS SELECT * FROM {SCHEMA}.{PREDICTIONS_TABLE}
        """)
        print(f"Feature table {online_table_name} created as online store!")
    except Exception as e2:
        print(f"Alternative approach also failed: {e2}")

# COMMAND ----------

print("\\n=== Pipeline Complete ===")
print(f"Feature Group: {SCHEMA}.{FEATURE_GROUP}")
print(f"Training Dataset: {SCHEMA}.{TRAINING_DATASET}")
print(f"Model: {SCHEMA}.{MODEL_NAME}")
print(f"Predictions Table: {SCHEMA}.{PREDICTIONS_TABLE}")
print(f"Validation ROC AUC: {roc_auc:.4f}")
'''
    
    # Write notebook to workspace
    notebook_path = f"{WORKSPACE_PATH}/fraud_pipeline.ipynb"
    # Convert to proper notebook format
    notebook_json = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# MAGIC %md\n", "# MAGIC ## Credit Card Fraud Detection Pipeline\n", "# MAGIC \n", "# MAGIC This notebook performs:\n", "# MAGIC 1. Feature engineering for fraud detection\n", "# MAGIC 2. Training dataset assembly\n", "# MAGIC 3. Model training and registration\n", "# MAGIC 4. Scoring of new transactions\n"]
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
    
    # Write notebook using workspace API
    wc.workspace.upload(notebook_path, json.dumps(notebook_json).encode('utf-8'), overwrite=True)
    
    # Step 3: Create and run a job to execute the notebook
    print("Creating job to run the notebook...")
    
    job_name = f"{PREFIX}_fraud_pipeline"
    
    # First, let's try to create a cluster
    cluster_name = f"{PREFIX}_cluster"
    
    # Create a simple job that runs the notebook
    job = wc.jobs.create(
        name=job_name,
        tasks=[
            {
                "task_key": "run_fraud_pipeline",
                "notebook_task": {
                    "notebook_path": notebook_path
                },
                "existing_cluster_id": "1201-221128-234758"  # Use a default cluster if available
            }
        ]
    )
    
    print(f"Job created: {job.job_id}")
    
    # Run the job
    run = wc.jobs.run_now(job.job_id)
    print(f"Job run started: {run.run_id}")
    
    # Wait for completion
    run_result = wc.jobs.wait_get_run_job_terminated_or_skipped(run.run_id, timeout_seconds=3600)
    print(f"Job run status: {run_result.state.life_cycle_state}")
    
    if run_result.state.life_cycle_state == "TERMINATED":
        print("Job completed successfully!")
    else:
        print(f"Job failed with state: {run_result.state.life_cycle_state}")
        print(f"Error: {run_result.state.result_state}")
    
    print("Pipeline setup complete!")

if __name__ == "__main__":
    main()
