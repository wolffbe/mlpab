#!/usr/bin/env python3
"""
Run the fraud detection pipeline on Databricks.
This script creates a notebook and runs it as a job.
"""
import os
import json
import time
from databricks.sdk import WorkspaceClient

# Configuration
SCHEMA = os.environ.get('MLPAB_DATABRICKS_SCHEMA', 'workspace.mlpabf21a49')
PREFIX = os.environ.get('MLPAB_DATABRICKS_PREFIX', 'mlpabf21a49')

FEATURE_GROUP = 'cctxn0802aa'
TRAINING_DATASET = 'cctd0802aa'
MODEL_NAME = 'ccmodel0802aa'
PREDICTIONS_TABLE = 'ccpred0802aa'

# Get user info
wc = WorkspaceClient()
current_user = wc.current_user.me()
USER_HOME = f'/Users/{current_user.user_name}'
WORKSPACE_PATH = f'{USER_HOME}/{PREFIX}'

def main():
    print("Starting fraud detection pipeline...")
    print(f"Schema: {SCHEMA}")
    print(f"Workspace path: {WORKSPACE_PATH}")
    
    # Step 1: Upload data files to DBFS
    print("\n=== Step 1: Uploading data files ===")
    dbfs_data_path = f'{WORKSPACE_PATH}/data'
    
    # Create directory
    wc.dbfs.mkdirs(f'dbfs:/{dbfs_data_path}')
    
    # Upload files
    wc.dbfs.upload(f'dbfs:/{dbfs_data_path}/transactions.csv', 'data/transactions.csv', overwrite=True)
    wc.dbfs.upload(f'dbfs:/{dbfs_data_path}/score_transactions.csv', 'data/score_transactions.csv', overwrite=True)
    print("Data files uploaded successfully!")
    
    # Step 2: Create the notebook
    print("\n=== Step 2: Creating notebook ===")
    
    notebook_content = f'''# Databricks notebook source
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

print(f"Schema: {{SCHEMA}}")
print(f"Workspace path: {{WORKSPACE_PATH}}")

# COMMAND ----------

# MAGIC %md ### Step 1: Load Data

# COMMAND ----------

print("Loading training data...")
train_df = spark.read.csv(f"{{WORKSPACE_PATH}}/data/transactions.csv", header=True, inferSchema=True)
print(f"Training data: {{train_df.count()}} rows, {{len(train_df.columns)}} columns")

print("Loading scoring data...")
score_df = spark.read.csv(f"{{WORKSPACE_PATH}}/data/score_transactions.csv", header=True, inferSchema=True)
print(f"Scoring data: {{score_df.count()}} rows, {{len(score_df.columns)}} columns")

# COMMAND ----------

# MAGIC %md ### Step 2: Feature Engineering

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import FloatType
import math

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

# Card statistics
card_stats = train_df.groupBy("cc_num").agg(
    F.avg("lat").alias("card_mean_lat"),
    F.avg("long").alias("card_mean_long"),
    F.avg("amount").alias("card_avg_amount"),
    F.stddev("amount").alias("card_std_amount"),
    F.count("*").alias("card_txn_count")
)

# Merchant and category statistics
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
fg_table_name = f"{{SCHEMA}}.{{FEATURE_GROUP}}"
print(f"Creating feature group: {{fg_table_name}}")
train_df.select(*feature_columns).write.mode("overwrite").saveAsTable(fg_table_name)
print(f"Feature group {{FEATURE_GROUP}} created!")

# COMMAND ----------

# MAGIC %md ### Step 4: Create Training Dataset

# COMMAND ----------

td_table_name = f"{{SCHEMA}}.{{TRAINING_DATASET}}"
print(f"Creating training dataset: {{td_table_name}}")
train_df.select(*feature_columns, 'is_fraud').write.mode("overwrite").saveAsTable(td_table_name)
print(f"Training dataset {{TRAINING_DATASET}} created!")

# Check fraud rate
fraud_rate = train_df.filter("is_fraud = 1").count() / train_df.count()
print(f"Fraud rate: {{fraud_rate:.4f}}")

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
print(f"ML data: {{ml_df.count()}} rows")

# Categorical and numeric columns
categorical_cols = ['merchant', 'category']
numeric_cols = [c for c in feature_columns if c not in categorical_cols + ['transaction_id', 'cc_num', 'datetime']]

print(f"Categorical cols: {{categorical_cols}}")
print(f"Numeric cols: {{numeric_cols}}")

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
                            numTrees=200, maxDepth=6, seed=42, subsamplingRate=0.8)

# Pipeline
pipeline = Pipeline(stages=indexers + [encoder, assembler, scaler, rf])

# Split data
(train_data, val_data) = ml_df.randomSplit([0.8, 0.2], seed=42)
print(f"Training: {{train_data.count()}}, Validation: {{val_data.count()}}")

# Train model
print("Training model...")
model = pipeline.fit(train_data)

# Predict on validation
predictions = model.transform(val_data)

# Evaluate
evaluator = BinaryClassificationEvaluator(labelCol="is_fraud", metricName="areaUnderROC")
roc_auc = evaluator.evaluate(predictions)
print(f"Validation ROC AUC: {{roc_auc:.4f}}")

# Log model with MLflow
print("Logging model to MLflow...")
with mlflow.start_run() as run:
    mlflow.log_param("num_trees", 200)
    mlflow.log_param("max_depth", 6)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.spark.log_model(model, MODEL_NAME)
    signature = infer_signature(val_data.select("features"), predictions.select("prediction"))
    mlflow.set_signature(signature)
    run_id = run.info.run_id
    print(f"MLflow run ID: {{run_id}}")

# COMMAND ----------

# MAGIC %md ### Step 6: Register Model

# COMMAND ----------

print("Registering model in Unity Catalog...")
model_uri = f"runs:/{{run_id}}/model"
mlflow.set_registry_uri("databricks-uc")
result = mlflow.register_model(
    model_uri=model_uri,
    name=f"{{SCHEMA}}.{{MODEL_NAME}}"
)
print(f"Model registered: {{result.name}}, version: {{result.version}}")

# COMMAND ----------

# MAGIC %md ### Step 7: Score Transactions

# COMMAND ----------

print("Scoring transactions...")
# Apply same feature engineering to score data
score_ml_df = score_df.na.drop()
print(f"Scoring data: {{score_ml_df.count()}} rows")

# Apply the same transformations and predict
score_predictions = model.transform(score_ml_df)

# Extract predictions
predictions_df = score_predictions.select(
    "transaction_id",
    F.col("probability")[1].alias("fraud_probability")
)

print(f"Predictions: {{predictions_df.count()}} rows")
predictions_df.limit(10).show()

# COMMAND ----------

# MAGIC %md ### Step 8: Create Predictions Table

# COMMAND ----------

predictions_table_name = f"{{SCHEMA}}.{{PREDICTIONS_TABLE}}"
print(f"Creating predictions table: {{predictions_table_name}}")
predictions_df.write.mode("overwrite").saveAsTable(predictions_table_name)
print(f"Predictions table {{PREDICTIONS_TABLE}} created!")

# Verify predictions
predictions_df.agg({"fraud_probability": "min"}, {"fraud_probability": "max"}, {"fraud_probability": "avg"}).show()

# COMMAND ----------

# MAGIC %md ### Step 9: Create Online Table

# COMMAND ----------

print("Creating online table for low-latency lookup...")
try:
    spark.sql(f"""
    CREATE ONLINE TABLE IF NOT EXISTS {{SCHEMA}}.{{PREDICTIONS_TABLE}}_online
    AS SELECT transaction_id, fraud_probability FROM {{SCHEMA}}.{{PREDICTIONS_TABLE}}
    """)
    print("Online table created!")
except Exception as e:
    print(f"Online table creation failed: {{e}}")
    # Try creating as a feature table
    try:
        spark.sql(f"""
        CREATE FEATURE TABLE IF NOT EXISTS {{SCHEMA}}.{{PREDICTIONS_TABLE}}_online
        AS SELECT transaction_id, fraud_probability FROM {{SCHEMA}}.{{PREDICTIONS_TABLE}}
        """)
        print("Feature table created as online store!")
    except Exception as e2:
        print(f"Feature table creation also failed: {{e2}}")

# COMMAND ----------

print("\\n=== Pipeline Complete ===")
print(f"Feature Group: {{SCHEMA}}.{{FEATURE_GROUP}}")
print(f"Training Dataset: {{SCHEMA}}.{{TRAINING_DATASET}}")
print(f"Model: {{SCHEMA}}.{{MODEL_NAME}}")
print(f"Predictions Table: {{SCHEMA}}.{{PREDICTIONS_TABLE}}")
print(f"Validation ROC AUC: {{roc_auc:.4f}}")
'''
    
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
    
    wc.workspace.upload(notebook_path, json.dumps(notebook_json).encode('utf-8'), overwrite=True)
    print(f"Notebook created at: {notebook_path}")
    
    # Step 3: Create a cluster
    print("\n=== Step 3: Creating cluster ===")
    
    cluster_name = f"{PREFIX}_fraud_cluster"
    try:
        cluster = wc.clusters.create(
            cluster_name=cluster_name,
            spark_version="14.3.x-scala2.12",
            node_type_id="Standard_DS3_v2",
            num_workers=2,
            policy_id="00172F64B1D6A0FB",  # Job Compute policy
            autotermination_minutes=30
        )
        print(f"Cluster created: {cluster.cluster_id}")
        
        # Wait for cluster to be running
        print("Waiting for cluster to start...")
        while True:
            cluster_info = wc.clusters.get(cluster.cluster_id)
            if cluster_info.state == "RUNNING":
                print("Cluster is running!")
                break
            elif cluster_info.state == "ERROR":
                print(f"Cluster failed: {cluster_info.state_message}")
                raise Exception(f"Cluster failed: {cluster_info.state_message}")
            time.sleep(10)
        
        cluster_id = cluster.cluster_id
    except Exception as e:
        print(f"Cluster creation failed: {e}")
        print("Trying with existing cluster or different approach...")
        # Try to use an existing cluster or create with different settings
        try:
            cluster = wc.clusters.create(
                cluster_name=cluster_name,
                spark_version="14.3.x-scala2.12",
                node_type_id="Standard_DS3_v2",
                num_workers=1,
                autotermination_minutes=30
            )
            print(f"Cluster created with simplified config: {cluster.cluster_id}")
            
            # Wait for cluster
            while True:
                cluster_info = wc.clusters.get(cluster.cluster_id)
                if cluster_info.state == "RUNNING":
                    print("Cluster is running!")
                    break
                elif cluster_info.state == "ERROR":
                    raise Exception(f"Cluster failed: {cluster_info.state_message}")
                time.sleep(10)
            
            cluster_id = cluster.cluster_id
        except Exception as e2:
            print(f"All cluster creation attempts failed: {e2}")
            print("Will try to use serverless warehouse instead...")
            cluster_id = None
    
    # Step 4: Create and run job
    print("\n=== Step 4: Creating and running job ===")
    
    job_name = f"{PREFIX}_fraud_pipeline"
    
    if cluster_id:
        # Use the created cluster
        job = wc.jobs.create(
            name=job_name,
            tasks=[
                {
                    "task_key": "run_fraud_pipeline",
                    "notebook_task": {
                        "notebook_path": notebook_path
                    },
                    "existing_cluster_id": cluster_id
                }
            ]
        )
    else:
        # Use serverless warehouse
        job = wc.jobs.create(
            name=job_name,
            tasks=[
                {
                    "task_key": "run_fraud_pipeline",
                    "notebook_task": {
                        "notebook_path": notebook_path
                    },
                    "warehouse_id": WAREHOUSE_ID
                }
            ]
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
        
        time.sleep(10)
    
    # Get job output
    try:
        output = wc.jobs.get_run_output(run.run_id)
        print(f"Job output: {output}")
    except Exception as e:
        print(f"Could not get job output: {e}")
    
    print("\n=== Pipeline setup complete ===")
    print(f"Feature Group: {SCHEMA}.{FEATURE_GROUP}")
    print(f"Training Dataset: {SCHEMA}.{TRAINING_DATASET}")
    print(f"Model: {SCHEMA}.{MODEL_NAME}")
    print(f"Predictions Table: {SCHEMA}.{PREDICTIONS_TABLE}")

if __name__ == "__main__":
    main()
