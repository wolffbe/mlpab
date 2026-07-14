# Databricks notebook source
# MAGIC %md
# MAGIC # Credit Card Fraud Detection Pipeline

# COMMAND ----------

import os
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import mlflow
import mlflow.spark
import math

# Get the schema from environment
schema_name = os.getenv('MLPAB_DATABRICKS_SCHEMA', 'workspace.mlpabfcf9c1')
prefix = os.getenv('MLPAB_DATABRICKS_PREFIX', 'mlpabfcf9c1')

# Parse schema
catalog, schema = schema_name.split('.')

print(f"Catalog: {catalog}, Schema: {schema}")

# COMMAND ----------

# MAGIC %md ## Step 1: Load Data

# COMMAND ----------

# Read the CSV files from workspace
# In Databricks, workspace files are accessible via dbfs:/user/...
transactions_path = f"dbfs:/user/benedict@hopsworks.ai/{prefix}/data/transactions.csv"
score_path = f"dbfs:/user/benedict@hopsworks.ai/{prefix}/data/score_transactions.csv"

print(f"Reading transactions from: {transactions_path}")
print(f"Reading score transactions from: {score_path}")

# Use Spark to read CSV files
df_transactions = spark.read.csv(transactions_path, header=True, inferSchema=True)
df_score = spark.read.csv(score_path, header=True, inferSchema=True)

# Convert datetime string to timestamp
df_transactions = df_transactions.withColumn("datetime", F.to_timestamp("datetime", "yyyy-MM-dd'T'HH:mm:ss'Z'"))
df_score = df_score.withColumn("datetime", F.to_timestamp("datetime", "yyyy-MM-dd'T'HH:mm:ss'Z'"))

print(f"Transactions count: {df_transactions.count()}")
print(f"Score transactions count: {df_score.count()}")
print(f"Transactions columns: {df_transactions.columns}")
print(f"Fraud rate: {df_transactions.filter('is_fraud = 1').count() / df_transactions.count():.4f}")

# COMMAND ----------

# MAGIC %md ## Step 2: Feature Engineering

# COMMAND ----------

def haversine_udf(lat1, lon1, lat2, lon2):
    """Calculate haversine distance between two points in km"""
    try:
        R = 6371.0  # Earth radius in km
        lat1_rad = math.radians(float(lat1)) if lat1 is not None else 0.0
        lon1_rad = math.radians(float(lon1)) if lon1 is not None else 0.0
        lat2_rad = math.radians(float(lat2)) if lat2 is not None else 0.0
        lon2_rad = math.radians(float(lon2)) if lon2 is not None else 0.0
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a)) if a < 1 else 0.0
        return R * c
    except:
        return 0.0

# Register UDF
haversine = F.udf(haversine_udf, DoubleType())

# Feature engineering
window_spec = Window.partitionBy("cc_num")
window_spec_time = Window.partitionBy("cc_num").orderBy("datetime")

# Extract time features
df_transactions = df_transactions.withColumn("hour_of_day", F.hour("datetime"))
df_transactions = df_transactions.withColumn("day_of_week", F.dayofweek("datetime"))
df_transactions = df_transactions.withColumn("day_of_month", F.dayofmonth("datetime"))
df_transactions = df_transactions.withColumn("month", F.month("datetime"))

# Amount features
df_transactions = df_transactions.withColumn("amount_log", F.log1p("amount"))
df_transactions = df_transactions.withColumn("amount_mean", F.mean("amount").over(window_spec))
df_transactions = df_transactions.withColumn("amount_std", F.stddev("amount").over(window_spec))
df_transactions = df_transactions.withColumn("amount_zscore", 
    (F.col("amount") - F.col("amount_mean")) / (F.col("amount_std") + 1e-6))

# Time since last transaction per card (in hours)
df_transactions = df_transactions.withColumn("time_since_last_txn", 
    F.coalesce((F.unix_timestamp("datetime") - F.unix_timestamp(F.lag("datetime").over(window_spec_time))) / 3600.0, F.lit(999.0)))

# Transaction count per card
df_transactions = df_transactions.withColumn("txn_count_per_card", 
    F.count("transaction_id").over(window_spec))

# Merchant and category frequency per card
df_transactions = df_transactions.withColumn("merchant_count_per_card", 
    F.countDistinct("merchant").over(window_spec))
df_transactions = df_transactions.withColumn("category_count_per_card", 
    F.countDistinct("category").over(window_spec))

# Calculate card's usual location (mean lat/long)
df_transactions = df_transactions.withColumn("card_lat_mean", 
    F.mean("lat").over(window_spec))
df_transactions = df_transactions.withColumn("card_long_mean", 
    F.mean("long").over(window_spec))

# Distance from usual location
df_transactions = df_transactions.withColumn("distance_from_usual_km", 
    haversine(F.col("lat"), F.col("long"), F.col("card_lat_mean"), F.col("card_long_mean")))

# Distance features
df_transactions = df_transactions.withColumn("distance_std", 
    F.stddev("distance_from_usual_km").over(window_spec))
df_transactions = df_transactions.withColumn("distance_mean", 
    F.mean("distance_from_usual_km").over(window_spec))
df_transactions = df_transactions.withColumn("distance_zscore", 
    (F.col("distance_from_usual_km") - F.col("distance_mean")) / (F.col("distance_std") + 1e-6))

# Is weekend
df_transactions = df_transactions.withColumn("is_weekend", 
    F.when(F.col("day_of_week") >= 6, 1.0).otherwise(0.0))

# Is night (10 PM to 6 AM)
df_transactions = df_transactions.withColumn("is_night", 
    F.when((F.col("hour_of_day") >= 22) | (F.col("hour_of_day") < 6), 1.0).otherwise(0.0))

# Category encoding using StringIndexer
category_indexer = StringIndexer(inputCol="category", outputCol="category_index")
df_transactions = category_indexer.fit(df_transactions).transform(df_transactions)

print(f"Feature engineering complete")

# COMMAND ----------

# MAGIC %md ## Step 3: Create Feature Table (Feature Group) and Training Dataset

# COMMAND ----------

# Write to feature table cctxn015310
feature_table_name = f"{catalog}.{schema}.cctxn015310"
df_transactions.write.mode("overwrite").saveAsTable(feature_table_name)

print(f"Feature table created: {feature_table_name}")

# Also create training dataset table cctd015310
training_table_name = f"{catalog}.{schema}.cctd015310"
df_transactions.write.mode("overwrite").saveAsTable(training_table_name)

print(f"Training dataset created: {training_table_name}")

# COMMAND ----------

# MAGIC %md ## Step 4: Train Classifier

# COMMAND ----------

# Read back the feature table
spark_df = spark.table(feature_table_name)

# Select features for training
feature_cols = [
    'amount', 'amount_log', 'amount_mean', 'amount_std', 'amount_zscore',
    'time_since_last_txn',
    'txn_count_per_card',
    'merchant_count_per_card', 'category_count_per_card',
    'card_lat_mean', 'card_long_mean', 'distance_from_usual_km',
    'distance_std', 'distance_mean', 'distance_zscore',
    'hour_of_day', 'day_of_week', 'day_of_month', 'month',
    'is_weekend', 'is_night', 'category_index'
]

# Replace nulls with 0 for numeric columns
for c in feature_cols:
    spark_df = spark_df.withColumn(c, F.coalesce(F.col(c), F.lit(0.0)))

# Prepare data for MLlib
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
assembler_model = assembler.fit(spark_df)
df_assembled = assembler_model.transform(spark_df)

# Split into train and validation
train_data, val_data = df_assembled.randomSplit([0.8, 0.2], seed=42)

print(f"Train count: {train_data.count()}, Validation count: {val_data.count()}")

# Train Random Forest classifier
rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="is_fraud",
    numTrees=100,
    maxDepth=10,
    minInstancesPerNode=5,
    seed=42,
    subsamplingRate=0.8,
    impurity="gini"
)

rf_model = rf.fit(train_data)

# Predict on validation
val_predictions = rf_model.transform(val_data)

# Evaluate
evaluator = BinaryClassificationEvaluator(
    labelCol="is_fraud",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

val_auc = evaluator.evaluate(val_predictions)
print(f"Validation ROC AUC: {val_auc:.4f}")

# COMMAND ----------

# MAGIC %md ## Step 5: Register Model

# COMMAND ----------

# Set MLflow tracking
mlflow.set_experiment(f"/Users/benedict@hopsworks.ai/{prefix}/ccfraud_experiment")

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("num_trees", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("min_instances_per_node", 5)
    mlflow.log_param("seed", 42)
    
    # Log metrics
    mlflow.log_metric("val_roc_auc", val_auc)
    
    # Log model
    mlflow.spark.log_model(rf_model, "model")
    
    # Register model in Unity Catalog
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
    mv = mlflow.register_model(
        model_uri=model_uri,
        name=f"{catalog}.{schema}.ccmodel015310"
    )
    
    print(f"Model registered: {mv.name}")
    print(f"Model version: {mv.version}")

# COMMAND ----------

# MAGIC %md ## Step 6: Score Transactions

# COMMAND ----------

# Apply same feature engineering to score data
# Extract time features
df_score = df_score.withColumn("hour_of_day", F.hour("datetime"))
df_score = df_score.withColumn("day_of_week", F.dayofweek("datetime"))
df_score = df_score.withColumn("day_of_month", F.dayofmonth("datetime"))
df_score = df_score.withColumn("month", F.month("datetime"))

# Amount features
df_score = df_score.withColumn("amount_log", F.log1p("amount"))

# Get card statistics from training data
card_stats = spark_df.select("cc_num", "amount_mean", "amount_std", "card_lat_mean", "card_long_mean", "merchant_count_per_card", "category_count_per_card", "distance_std", "distance_mean", "txn_count_per_card").distinct()

# Join with score data
df_score = df_score.join(card_stats, on="cc_num", how="left")

# Fill nulls (for new cards)
for c in ["amount_mean", "amount_std", "card_lat_mean", "card_long_mean", "merchant_count_per_card", "category_count_per_card", "distance_std", "distance_mean", "txn_count_per_card"]:
    df_score = df_score.withColumn(c, F.coalesce(F.col(c), F.lit(0.0)))

# Calculate derived features
df_score = df_score.withColumn("amount_zscore", 
    (F.col("amount") - F.col("amount_mean")) / (F.col("amount_std") + 1e-6))

df_score = df_score.withColumn("time_since_last_txn", F.lit(999.0))

# Distance from usual location
df_score = df_score.withColumn("distance_from_usual_km", 
    haversine(F.col("lat"), F.col("long"), F.col("card_lat_mean"), F.col("card_long_mean")))

df_score = df_score.withColumn("distance_zscore", 
    (F.col("distance_from_usual_km") - F.col("distance_mean")) / (F.col("distance_std") + 1e-6))

# Is weekend and night
df_score = df_score.withColumn("is_weekend", F.lit(0.0))
df_score = df_score.withColumn("is_night", F.lit(0.0))

# Category encoding - use the same indexer
category_indexer_model = category_indexer.fit(df_score)
df_score = category_indexer_model.transform(df_score)

# Replace nulls in feature columns
for c in feature_cols:
    df_score = df_score.withColumn(c, F.coalesce(F.col(c), F.lit(0.0)))

# Assemble features
df_score_assembled = assembler_model.transform(df_score)

# Predict
score_predictions = rf_model.transform(df_score_assembled)

# Create predictions DataFrame
predictions_df = score_predictions.select(
    "transaction_id",
    F.col("probability")[1].alias("fraud_probability")
)

print(f"Predictions count: {predictions_df.count()}")

# COMMAND ----------

# MAGIC %md ## Step 7: Create Predictions Table

# COMMAND ----------

# Write to predictions table ccpred015310
predictions_table_name = f"{catalog}.{schema}.ccpred015310"
predictions_df.write.mode("overwrite").saveAsTable(predictions_table_name)

print(f"Predictions table created: {predictions_table_name}")

# COMMAND ----------

# MAGIC %md ## Step 8: Publish for Online Lookup

# COMMAND ----------

# Publish feature table for online lookup
try:
    from databricks.feature_store import FeatureStoreClient
    fs = FeatureStoreClient()
    
    # Publish feature table
    fs.publish_table(
        name=feature_table_name,
        online=True
    )
    print(f"Published {feature_table_name} for online lookup")
    
    # Publish predictions table
    fs.publish_table(
        name=predictions_table_name,
        online=True
    )
    print(f"Published {predictions_table_name} for online lookup")
except Exception as e:
    print(f"Could not publish online tables: {e}")
    print("Tables are still available as offline tables")

# COMMAND ----------

print("\n=== PIPELINE COMPLETE ===")
print(f"Feature group: {catalog}.{schema}.cctxn015310")
print(f"Training dataset: {catalog}.{schema}.cctd015310")
print(f"Model: {catalog}.{schema}.ccmodel015310")
print(f"Predictions table: {catalog}.{schema}.ccpred015310")
print(f"Validation ROC AUC: {val_auc:.4f}")
