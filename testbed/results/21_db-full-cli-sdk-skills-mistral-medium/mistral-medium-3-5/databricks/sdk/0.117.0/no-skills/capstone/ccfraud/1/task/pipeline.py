#!/usr/bin/env python3
"""
Full FTI pipeline for credit card fraud detection.
Creates feature group cctxn015310, training dataset cctd015310, 
model ccmodel015310, and predictions table ccpred015310.
"""

import os
import json
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog, ml, compute

# Environment variables
SCHEMA = os.environ.get("MLPAB_DATABRICKS_SCHEMA", "workspace.mlpab958e4d")
PREFIX = os.environ.get("MLPAB_DATABRICKS_PREFIX", "mlpab958e4d")

# Object names
FEATURE_GROUP_NAME = "cctxn015310"
TRAINING_DATASET_NAME = "cctd015310"
MODEL_NAME = "ccmodel015310"
PREDICTIONS_TABLE_NAME = "ccpred015310"

# Feature group full name
FG_FULL_NAME = f"{SCHEMA}.{FEATURE_GROUP_NAME}"
TD_FULL_NAME = f"{SCHEMA}.{TRAINING_DATASET_NAME}"
MODEL_FULL_NAME = f"{SCHEMA}.{MODEL_NAME}"
PRED_FULL_NAME = f"{SCHEMA}.{PREDICTIONS_TABLE_NAME}"

print(f"Schema: {SCHEMA}")
print(f"Prefix: {PREFIX}")
print(f"Feature Group: {FG_FULL_NAME}")
print(f"Training Dataset: {TD_FULL_NAME}")
print(f"Model: {MODEL_FULL_NAME}")
print(f"Predictions: {PRED_FULL_NAME}")

# Initialize workspace client
w = WorkspaceClient()

# Create the schema if it doesn't exist
try:
    catalog_client = w.catalog
    schemas = catalog_client.list_schemas(catalog_name="workspace")
    schema_names = [s.name for s in schemas]
    
    if SCHEMA.split(".")[1] not in schema_names:
        print(f"Creating schema {SCHEMA}")
        catalog_client.create_schema(
            catalog_name="workspace",
            schema_name=SCHEMA.split(".")[1],
            comment="MLPAB schema for fraud detection pipeline"
        )
    else:
        print(f"Schema {SCHEMA} already exists")
except Exception as e:
    print(f"Error checking/creating schema: {e}")

print("Step 1: Upload data files to DBFS")
# Upload the CSV files to DBFS
try:
    # Upload transactions.csv
    with open("data/transactions.csv", "rb") as f:
        w.dbfs.upload("dbfs:/FileStore/transactions.csv", f, overwrite=True)
    print("Uploaded transactions.csv")
    
    # Upload score_transactions.csv  
    with open("data/score_transactions.csv", "rb") as f:
        w.dbfs.upload("dbfs:/FileStore/score_transactions.csv", f, overwrite=True)
    print("Uploaded score_transactions.csv")
except Exception as e:
    print(f"Error uploading files: {e}")

print("\nStep 2: Create feature engineering notebook")

# Create a notebook for feature engineering and model training
notebook_content = """# Credit Card Fraud Detection Pipeline

# Step 1: Load and prepare data
transactions_df = spark.read.csv("dbfs:/FileStore/transactions.csv", header=True, inferSchema=True)
score_df = spark.read.csv("dbfs:/FileStore/score_transactions.csv", header=True, inferSchema=True)

print(f"Transactions: {transactions_df.count()}")
print(f"Score transactions: {score_df.count()}")

# Step 2: Feature Engineering
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import pyspark.sql.types as T

# Convert datetime to timestamp
transactions_df = transactions_df.withColumn("datetime", F.to_timestamp("datetime"))
score_df = score_df.withColumn("datetime", F.to_timestamp("datetime"))

# Feature 1: Hour of day
transactions_df = transactions_df.withColumn("hour_of_day", F.hour("datetime"))
score_df = score_df.withColumn("hour_of_day", F.hour("datetime"))

# Feature 2: Day of week (0=Monday, 6=Sunday)
transactions_df = transactions_df.withColumn("day_of_week", F.dayofweek("datetime") - 1)
score_df = score_df.withColumn("day_of_week", F.dayofweek("datetime") - 1)

# Feature 3: Amount features
transactions_df = transactions_df.withColumn("amount_log", F.log(F.col("amount") + 1))
score_df = score_df.withColumn("amount_log", F.log(F.col("amount") + 1))

# Feature 4: Transaction velocity per card (count in last 24 hours)
window_spec_24h = Window.partitionBy("cc_num").orderBy("datetime").rowsBetween(-24, 0)

# For training data - we need to compute this carefully
# First, let's compute time differences and rolling counts
from pyspark.sql.functions import lag, when

# Add row number within each card
txn_window = Window.partitionBy("cc_num").orderBy("datetime")
transactions_df = transactions_df.withColumn("row_num", F.row_number().over(txn_window))

# Self-join approach for velocity features
# Create a temporary view for SQL operations
transactions_df.createOrReplaceTempView("transactions")

# Feature: Count of transactions per card in last 24 hours
velocity_query = """
WITH ranked_txns AS (
  SELECT *,
         ROW_NUMBER() OVER (PARTITION BY cc_num ORDER BY datetime) as rn
  FROM transactions
),
velocity_features AS (
  SELECT 
    t1.transaction_id,
    t1.cc_num,
    t1.datetime,
    COUNT(t2.transaction_id) - 1 as txn_count_24h
  FROM ranked_txns t1
  LEFT JOIN ranked_txns t2 
    ON t1.cc_num = t2.cc_num 
    AND t2.datetime >= t1.datetime - INTERVAL 24 HOURS
    AND t2.datetime < t1.datetime
  GROUP BY t1.transaction_id, t1.cc_num, t1.datetime
)
SELECT * FROM velocity_features
"""

velocity_df = spark.sql(velocity_query)
transactions_df = transactions_df.join(velocity_df, "transaction_id", "left")
transactions_df = transactions_df.fillna({"txn_count_24h": 0})

# Do the same for score data
score_df.createOrReplaceTempView("score_transactions")
score_velocity_query = """
WITH ranked_txns AS (
  SELECT *,
         ROW_NUMBER() OVER (PARTITION BY cc_num ORDER BY datetime) as rn
  FROM score_transactions
),
velocity_features AS (
  SELECT 
    t1.transaction_id,
    t1.cc_num,
    t1.datetime,
    COUNT(t2.transaction_id) - 1 as txn_count_24h
  FROM ranked_txns t1
  LEFT JOIN ranked_txns t2 
    ON t1.cc_num = t2.cc_num 
    AND t2.datetime >= t1.datetime - INTERVAL 24 HOURS
    AND t2.datetime < t1.datetime
  GROUP BY t1.transaction_id, t1.cc_num, t1.datetime
)
SELECT * FROM velocity_features
"""

score_velocity_df = spark.sql(score_velocity_query)
score_df = score_df.join(score_velocity_df, "transaction_id", "left")
score_df = score_df.fillna({"txn_count_24h": 0})

# Feature 5: Average transaction amount per card (from historical data)
card_stats = transactions_df.groupBy("cc_num").agg(
    F.avg("amount").alias("card_avg_amount"),
    F.stddev("amount").alias("card_std_amount"),
    F.count("*").alias("card_txn_count")
)

transactions_df = transactions_df.join(card_stats, "cc_num", "left")
score_df = score_df.join(card_stats, "cc_num", "left")

# Feature 6: Amount deviation from card average
transactions_df = transactions_df.withColumn(
    "amount_dev_from_avg", 
    F.when(F.col("card_avg_amount").isNull(), 0).otherwise(F.col("amount") / F.col("card_avg_amount"))
)
score_df = score_df.withColumn(
    "amount_dev_from_avg", 
    F.when(F.col("card_avg_amount").isNull(), 0).otherwise(F.col("amount") / F.col("card_avg_amount"))
)

# Feature 7: Merchant frequency for card
merchant_freq = transactions_df.groupBy("cc_num", "merchant").agg(
    F.count("*").alias("merchant_count")
)
merchant_freq = merchant_freq.groupBy("cc_num").agg(
    F.max("merchant_count").alias("max_merchant_freq"),
    F.countDistinct("merchant").alias("unique_merchants")
)

transactions_df = transactions_df.join(merchant_freq, "cc_num", "left")
score_df = score_df.join(merchant_freq, "cc_num", "left")

# Feature 8: Category encoding and frequency
category_freq = transactions_df.groupBy("cc_num", "category").agg(
    F.count("*").alias("category_count")
)
category_freq = category_freq.groupBy("cc_num").agg(
    F.max("category_count").alias("max_category_freq")
)

transactions_df = transactions_df.join(category_freq, "cc_num", "left")
score_df = score_df.join(category_freq, "cc_num", "left")

# Feature 9: Geo distance from card's usual location
# For each card, compute centroid of all its transactions
card_geo = transactions_df.groupBy("cc_num").agg(
    F.avg("lat").alias("card_avg_lat"),
    F.avg("long").alias("card_avg_long"),
    F.stddev("lat").alias("card_lat_std"),
    F.stddev("long").alias("card_long_std")
)

# Haversine distance function
def haversine(lat1, lon1, lat2, lon2):
    from math import radians, sin, cos, sqrt, atan2
    R = 6371  # Earth radius in km
    dLat = radians(lat2 - lat1)
    dLon = radians(lon2 - lon1)
    a = sin(dLat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dLon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType

haversine_udf = udf(haversine, FloatType())

# Join with card geo stats
transactions_df = transactions_df.join(card_geo, "cc_num", "left")
score_df = score_df.join(card_geo, "cc_num", "left")

# Compute distance from card's average location
transactions_df = transactions_df.withColumn(
    "geo_distance_from_avg",
    haversine_udf(
        F.col("lat"), F.col("long"),
        F.col("card_avg_lat"), F.col("card_avg_long")
    )
)
score_df = score_df.withColumn(
    "geo_distance_from_avg",
    haversine_udf(
        F.col("lat"), F.col("long"),
        F.col("card_avg_lat"), F.col("card_avg_long")
    )
)

# Fill nulls for cards with only one transaction
transactions_df = transactions_df.fillna({
    "card_avg_amount": 0,
    "card_std_amount": 0,
    "card_txn_count": 0,
    "max_merchant_freq": 0,
    "unique_merchants": 0,
    "max_category_freq": 0,
    "card_avg_lat": 0,
    "card_avg_long": 0,
    "card_lat_std": 0,
    "card_long_std": 0,
    "geo_distance_from_avg": 0
})

score_df = score_df.fillna({
    "card_avg_amount": 0,
    "card_std_amount": 0,
    "card_txn_count": 0,
    "max_merchant_freq": 0,
    "unique_merchants": 0,
    "max_category_freq": 0,
    "card_avg_lat": 0,
    "card_avg_long": 0,
    "card_lat_std": 0,
    "card_long_std": 0,
    "geo_distance_from_avg": 0
})

# Replace infinite values
transactions_df = transactions_df.replace(float('inf'), 0)
score_df = score_df.replace(float('inf'), 0)

print("Feature engineering complete")
print(f"Training features: {transactions_df.columns}")

# Step 3: Create Feature Group
from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()

# Define the feature table schema
feature_table = fs.create_table(
    name=FG_FULL_NAME,
    primary_keys=["transaction_id"],
    df=transactions_df,
    description="Credit card transaction features for fraud detection"
)

print(f"Created feature group: {FG_FULL_NAME}")

# Step 4: Create Training Dataset
# We need to select the features and label for training
training_features = [
    "amount", "hour_of_day", "day_of_week", "amount_log",
    "txn_count_24h", "card_avg_amount", "card_std_amount", 
    "card_txn_count", "amount_dev_from_avg", "max_merchant_freq",
    "unique_merchants", "max_category_freq", "geo_distance_from_avg",
    "lat", "long", "is_fraud"
]

# Create training dataset
training_df = transactions_df.select(training_features)

td = fs.create_table(
    name=TD_FULL_NAME,
    primary_keys=["transaction_id"],
    df=training_df,
    description="Training dataset for credit card fraud detection"
)

print(f"Created training dataset: {TD_FULL_NAME}")

# Step 5: Train Model
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Prepare data for ML
# Convert categorical columns
category_cols = ["merchant", "category"]

# Index categorical columns
indexers = [
    StringIndexer(inputCol=col, outputCol=f"{col}_idx", handleInvalid="keep").fit(training_df)
    for col in category_cols
]

# One-hot encode
encoder = OneHotEncoder(
    inputCols=[f"{col}_idx" for col in category_cols],
    outputCols=[f"{col}_encoded" for col in category_cols]
)

# Assemble features
numeric_cols = [
    "amount", "hour_of_day", "day_of_week", "amount_log",
    "txn_count_24h", "card_avg_amount", "card_std_amount", 
    "card_txn_count", "amount_dev_from_avg", "max_merchant_freq",
    "unique_merchants", "max_category_freq", "geo_distance_from_avg",
    "lat", "long"
]

encoded_cols = [f"{col}_encoded" for col in category_cols]
feature_cols = numeric_cols + encoded_cols

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)

# Random Forest classifier
rf = RandomForestClassifier(
    labelCol="is_fraud",
    featuresCol="features",
    numTrees=100,
    maxDepth=5,
    seed=42
)

# Build pipeline
pipeline = Pipeline(stages=[*indexers, encoder, assembler, rf])

# Split data
train_data, test_data = training_df.randomSplit([0.8, 0.2], seed=42)

# Train model
print("Training model...")
model = pipeline.fit(train_data)

# Evaluate
predictions = model.transform(test_data)
evaluator = BinaryClassificationEvaluator(
    labelCol="is_fraud",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

auc = evaluator.evaluate(predictions)
print(f"Test AUC: {auc}")

# Log metrics
from mlflow import log_metric
log_metric("auc", auc)

# Step 6: Register Model
import mlflow
from mlflow.models.signature import infer_signature

# Set MLflow experiment
mlflow.set_experiment(f"/Users/{spark.sql('SELECT current_user()').collect()[0][0]}/{PREFIX}/fraud_detection")

# Log model with signature
signature = infer_signature(train_data, model)

with mlflow.start_run():
    mlflow.log_metric("auc", auc)
    mlflow.spark.log_model(
        model,
        MODEL_NAME,
        signature=signature,
        registered_model_name=MODEL_FULL_NAME
    )
    
    # Register the model
    mv = mlflow.register_model(
        f"runs:/{mlflow.active_run().info.run_id}/model",
        MODEL_FULL_NAME
    )
    print(f"Registered model: {MODEL_FULL_NAME}")

print("Model training and registration complete")

# Step 7: Score the test transactions
print("Scoring test transactions...")

# Prepare score data with same features
score_feature_cols = [
    "amount", "hour_of_day", "day_of_week", "amount_log",
    "txn_count_24h", "card_avg_amount", "card_std_amount", 
    "card_txn_count", "amount_dev_from_avg", "max_merchant_freq",
    "unique_merchants", "max_category_freq", "geo_distance_from_avg",
    "lat", "long", "merchant", "category", "transaction_id"
]

score_data = score_df.select(score_feature_cols)

# Apply same transformations
score_predictions = model.transform(score_data)

# Extract fraud probability (from probability vector)
from pyspark.sql.functions import col, udf as spark_udf
from pyspark.sql.types import FloatType

# Get probability of class 1 (fraud)
get_prob = spark_udf(lambda v: float(v[1]) if v else 0.0, FloatType())

result_df = score_predictions.select(
    "transaction_id",
    get_prob("probability").alias("fraud_probability")
)

# Write predictions to feature table
fs.create_table(
    name=PRED_FULL_NAME,
    primary_keys=["transaction_id"],
    df=result_df,
    description="Fraud probability predictions for score transactions"
)

print(f"Created predictions table: {PRED_FULL_NAME}")
print(f"Predictions count: {result_df.count()}")

# Also make it available for low-latency lookup as an online table
# First, create a regular table
result_df.write.saveAsTable(PRED_FULL_NAME, mode="overwrite")

# Create online table for low-latency lookup
try:
    online_table_name = f"{PREFIX}_fraud_predictions_online"
    w.online_tables.create(
        name=online_table_name,
        source_table_name=PRED_FULL_NAME,
        comment="Online table for fraud probability lookup"
    )
    print(f"Created online table: {online_table_name}")
except Exception as e:
    print(f"Could not create online table: {e}")

print("Pipeline complete!")
"""

# Write the notebook
notebook_path = f"/Users/{w.current_user.me().display_name}/{PREFIX}/fraud_detection_pipeline"
try:
    w.workspace.upload(
        path=notebook_path,
        content=notebook_content,
        language="PYTHON",
        overwrite=True
    )
    print(f"Created notebook: {notebook_path}")
except Exception as e:
    print(f"Error creating notebook: {e}")

print("\nStep 3: Run the notebook as a job")

# Create a cluster for the job
cluster_name = f"{PREFIX}_fraud_cluster"
try:
    # Check if cluster exists
    clusters = w.clusters.list()
    cluster_exists = any(c.cluster_name == cluster_name for c in clusters)
    
    if not cluster_exists:
        # Create a small cluster
        cluster = w.clusters.create(
            cluster_name=cluster_name,
            node_type_id="Standard_DS3_v2",
            spark_version="14.3.x-scala2.12",
            num_workers=2,
            autoscale=compute.ClusterAutoscale(min_workers=2, max_workers=4),
            spark_conf={
                "spark.databricks.delta.preview.enabled": "true"
            }
        )
        print(f"Created cluster: {cluster_name}")
        
        # Wait for cluster to be ready
        print("Waiting for cluster to start...")
        time.sleep(60)
    else:
        print(f"Cluster {cluster_name} already exists")
except Exception as e:
    print(f"Error with cluster: {e}")

# Create and run a job
job_name = f"{PREFIX}_fraud_pipeline_job"
try:
    # Create the job
    job = w.jobs.create(
        name=job_name,
        tasks=[
            {
                "task_key": "fraud_pipeline",
                "notebook_task": {
                    "notebook_path": notebook_path
                },
                "existing_cluster_id": cluster_name
            }
        ]
    )
    print(f"Created job: {job_name} (ID: {job.job_id})")
    
    # Run the job
    run = w.jobs.run_now(job_id=job.job_id)
    print(f"Started job run: {run.run_id}")
    
    # Wait for job to complete
    print("Waiting for job to complete...")
    run_info = w.jobs.get_run(run_id=run.run_id)
    
    # Poll for completion
    max_wait = 3600  # 1 hour
    wait_time = 0
    while run_info.state.life_cycle_state not in ["TERMINATED", "SKIPPED"] and wait_time < max_wait:
        time.sleep(30)
        wait_time += 30
        run_info = w.jobs.get_run(run_id=run.run_id)
        print(f"Job status: {run_info.state.life_cycle_state}, {run_info.state.result_state}")
    
    if run_info.state.life_cycle_state == "TERMINATED":
        if run_info.state.result_state == "SUCCESS":
            print("Job completed successfully!")
        else:
            print(f"Job failed with result: {run_info.state.result_state}")
            print(f"Error: {run_info.state.state_message}")
    else:
        print("Job timed out")
        
except Exception as e:
    print(f"Error with job: {e}")

print("Pipeline setup complete!")
