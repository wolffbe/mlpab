#!/usr/bin/env python3
"""
Fraud FTI Pipeline:
1. Engineer fraud features into feature group `cctxne0b071`.
2. Assemble training dataset `cctde0b071`.
3. Train and register classifier `ccmodele0b071` with metrics.
4. Score `score_transactions.csv` into `ccprede0b071` with low-latency lookup.
"""
import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog, jobs, ml
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, unix_timestamp, lag, stddev, mean, count, lit, when
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline

# Initialize
w = WorkspaceClient()
spark = SparkSession.builder.getOrCreate()

# Config
schema_name = os.getenv("MLPAB_DATABRICKS_SCHEMA")  # workspace.mlpab21b96f
catalog_name = schema_name.split(".")[0]  # workspace
schema_name_only = schema_name.split(".")[1]  # mlpab21b96f
prefix = os.getenv("MLPAB_DATABRICKS_PREFIX")  # mlpab21b96f
user = os.getenv("USER")  # wolffbe

# Paths
volume_path = f"/Volumes/{catalog_name}/{schema_name_only}/{prefix}_fraud_data"
transactions_path = f"{volume_path}/transactions.csv"
score_path = f"{volume_path}/score_transactions.csv"

# Feature Group, Dataset, Model, and Predictions Table Names
feature_group_name = "cctxne0b071"
training_dataset_name = "cctde0b071"
model_name = "ccmodele0b071"
predictions_table_name = "ccprede0b071"

# Read data
print("Reading transactions data...")
transactions_df = spark.read.csv(transactions_path, header=True, inferSchema=True)
score_df = spark.read.csv(score_path, header=True, inferSchema=True)

# Feature Engineering
print("Engineering fraud features...")
# Convert datetime to timestamp
transactions_df = transactions_df.withColumn("timestamp", unix_timestamp(col("datetime")))

# Window for time-based features (e.g., transaction velocity, amount stats)
window_spec = Window.partitionBy("cc_num").orderBy("timestamp")

# Feature 1: Transaction velocity (count per card in last 1 hour)
transactions_df = transactions_df.withColumn(
    "txn_velocity_1h",
    count("transaction_id").over(window_spec.rangeBetween(-3600, 0))
)

# Feature 2: Amount statistics (mean and stddev per card in last 24 hours)
transactions_df = transactions_df.withColumn(
    "amount_mean_24h",
    mean("amount").over(window_spec.rangeBetween(-86400, 0))
)
transactions_df = transactions_df.withColumn(
    "amount_stddev_24h",
    stddev("amount").over(window_spec.rangeBetween(-86400, 0))
)

# Feature 3: Time since last transaction
transactions_df = transactions_df.withColumn(
    "time_since_last_txn",
    col("timestamp") - lag("timestamp").over(window_spec)
)

# Feature 4: Amount deviation from mean
transactions_df = transactions_df.withColumn(
    "amount_dev_from_mean",
    (col("amount") - col("amount_mean_24h")) / (col("amount_stddev_24h") + lit(1e-6))
)

# Feature 5: Geo distance from previous transaction (simplified)
transactions_df = transactions_df.withColumn(
    "prev_lat", lag("lat").over(window_spec)
).withColumn(
    "prev_long", lag("long").over(window_spec)
)
transactions_df = transactions_df.withColumn(
    "geo_distance",
    when(
        (col("prev_lat").isNotNull()) & (col("prev_long").isNotNull()),
        ((col("lat") - col("prev_lat"))**2 + (col("long") - col("prev_long"))**2)**0.5
    ).otherwise(lit(0.0))
)

# Drop intermediate columns
transactions_df = transactions_df.drop("prev_lat", "prev_long", "timestamp")

# Write to Feature Group (as a Delta table)
feature_group_full_name = f"{catalog_name}.{schema_name_only}.{feature_group_name}"
print(f"Writing feature group: {feature_group_full_name}")
transactions_df.write.format("delta").mode("overwrite").saveAsTable(feature_group_full_name)

# Assemble Training Dataset
print("Assembling training dataset...")
training_dataset_full_name = f"{catalog_name}.{schema_name_only}.{training_dataset_name}"
# Select features and label
feature_cols = [
    "txn_velocity_1h", "amount_mean_24h", "amount_stddev_24h",
    "time_since_last_txn", "amount_dev_from_mean", "geo_distance", "amount"
]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
training_df = assembler.transform(transactions_df)

# Split into train/test
(training_data, test_data) = training_df.randomSplit([0.8, 0.2], seed=42)

# Train Classifier
print("Training classifier...")
rf = RandomForestClassifier(labelCol="is_fraud", featuresCol="features", numTrees=100)
pipeline = Pipeline(stages=[rf])
model = pipeline.fit(training_data)

# Evaluate
print("Evaluating model...")
predictions = model.transform(test_data)
evaluator = BinaryClassificationEvaluator(labelCol="is_fraud", metricName="areaUnderROC")
roc_auc = evaluator.evaluate(predictions)
print(f"ROC AUC: {roc_auc}")

# Register Model
print("Registering model...")
model_path = f"/Users/{user}/{prefix}/fraud_model"
model.write().overwrite().save(model_path)

# Register in MLflow
model_full_name = f"{catalog_name}.{schema_name_only}.{model_name}"
w.model_versions.create(
    name=model_full_name,
    source=model_path,
    run_id=None,  # Not using MLflow runs here
)

# Score score_transactions.csv
print("Scoring transactions...")
# Apply the same feature engineering to score_df
score_df = score_df.withColumn("timestamp", unix_timestamp(col("datetime")))

# Reuse the same window and feature logic
score_df = score_df.withColumn(
    "txn_velocity_1h",
    count("transaction_id").over(window_spec.rangeBetween(-3600, 0))
)
score_df = score_df.withColumn(
    "amount_mean_24h",
    mean("amount").over(window_spec.rangeBetween(-86400, 0))
)
score_df = score_df.withColumn(
    "amount_stddev_24h",
    stddev("amount").over(window_spec.rangeBetween(-86400, 0))
)
score_df = score_df.withColumn(
    "time_since_last_txn",
    col("timestamp") - lag("timestamp").over(window_spec)
)
score_df = score_df.withColumn(
    "amount_dev_from_mean",
    (col("amount") - col("amount_mean_24h")) / (col("amount_stddev_24h") + lit(1e-6))
)
score_df = score_df.withColumn(
    "prev_lat", lag("lat").over(window_spec)
).withColumn(
    "prev_long", lag("long").over(window_spec)
)
score_df = score_df.withColumn(
    "geo_distance",
    when(
        (col("prev_lat").isNotNull()) & (col("prev_long").isNotNull()),
        ((col("lat") - col("prev_lat"))**2 + (col("long") - col("prev_long"))**2)**0.5
    ).otherwise(lit(0.0))
)

# Drop intermediate columns
score_df = score_df.drop("prev_lat", "prev_long", "timestamp")

# Apply model
score_df = assembler.transform(score_df)
predictions_df = model.transform(score_df)

# Select transaction_id and fraud_probability
predictions_df = predictions_df.select(
    "transaction_id",
    col("probability").getItem(1).alias("fraud_probability")
)

# Write to Predictions Table
predictions_table_full_name = f"{catalog_name}.{schema_name_only}.{predictions_table_name}"
print(f"Writing predictions table: {predictions_table_full_name}")
predictions_df.write.format("delta").mode("overwrite").saveAsTable(predictions_table_full_name)

# Enable Low-Latency Lookup
print("Enabling low-latency lookup...")
w.online_tables.create(
    name=predictions_table_full_name,
    spec=ml.OnlineTableSpec(
        primary_key_columns=["transaction_id"],
        source_table_full_name=predictions_table_full_name,
    ),
)

print("Pipeline complete.")
print(f"Feature Group: {feature_group_full_name}")
print(f"Training Dataset: {training_dataset_full_name}")
print(f"Model: {model_full_name}")
print(f"Predictions Table: {predictions_table_full_name}")
print(f"ROC AUC: {roc_auc}")