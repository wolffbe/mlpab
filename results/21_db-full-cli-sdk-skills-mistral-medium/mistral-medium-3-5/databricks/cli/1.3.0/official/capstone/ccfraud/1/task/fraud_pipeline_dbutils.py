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

# Use dbutils to read workspace files
# First, let's try to copy the CSV files to a temp location
import dbutils

# List workspace files
workspace_path = f"/Users/benedict@hopsworks.ai/{prefix}/data/"
print(f"Workspace path: {workspace_path}")

# Try to read the CSV files using dbutils
try:
    # Copy files to DBFS temp location
    dbutils.fs.cp(f"file:{workspace_path}transactions.csv", "dbfs:/tmp/transactions.csv", recurse=False)
    dbutils.fs.cp(f"file:{workspace_path}score_transactions.csv", "dbfs:/tmp/score_transactions.csv", recurse=False)
    print("Files copied to DBFS")
except Exception as e:
    print(f"Error copying files: {e}")
    # Try alternative approach
    pass

# Try to read from workspace directly
try:
    df_transactions = spark.read.csv(f"{workspace_path}transactions.csv", header=True, inferSchema=True)
    df_score = spark.read.csv(f"{workspace_path}score_transactions.csv", header=True, inferSchema=True)
    print(f"Read transactions: {df_transactions.count()}")
except Exception as e:
    print(f"Error reading from workspace: {e}")
    # Try from dbfs
    try:
        df_transactions = spark.read.csv("dbfs:/tmp/transactions.csv", header=True, inferSchema=True)
        df_score = spark.read.csv("dbfs:/tmp/score_transactions.csv", header=True, inferSchema=True)
        print(f"Read from dbfs/tmp: {df_transactions.count()}")
    except Exception as e2:
        print(f"Error reading from dbfs/tmp: {e2}")
        # Try from user workspace
        try:
            df_transactions = spark.read.csv(f"dbfs:/user/benedict@hopsworks.ai/{prefix}/data/transactions.csv", header=True, inferSchema=True)
            df_score = spark.read.csv(f"dbfs:/user/benedict@hopsworks.ai/{prefix}/data/score_transactions.csv", header=True, inferSchema=True)
            print(f"Read from dbfs/user: {df_transactions.count()}")
        except Exception as e3:
            print(f"Error reading from dbfs/user: {e3}")
            raise

# Convert datetime string to timestamp
df_transactions = df_transactions.withColumn("datetime", F.to_timestamp("datetime", "yyyy-MM-dd'T'HH:mm:ss'Z'"))
df_score = df_score.withColumn("datetime", F.to_timestamp("datetime", "yyyy-MM-dd'T'HH:mm:ss'Z'"))

print(f"Transactions count: {df_transactions.count()}")
print(f"Score transactions count: {df_score.count()}")
print(f"Fraud rate: {df_transactions.filter('is_fraud = 1').count() / df_transactions.count():.4f}")

# COMMAND ----------

# MAGIC %md ## Step 2: Feature Engineering

# COMMAND ----------

def haversine_udf(lat1, lon1, lat2, lon2):
    """Calculate haversine distance between two points in km"""
    try:
        R = 6371.0
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

haversine = F.udf(haversine_udf, DoubleType())

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

# Time since last transaction
df_transactions = df_transactions.withColumn("time_since_last_txn", 
    F.coalesce((F.unix_timestamp("datetime") - F.unix_timestamp(F.lag("datetime").over(window_spec_time))) / 3600.0, F.lit(999.0)))

# Transaction count per card
df_transactions = df_transactions.withColumn("txn_count_per_card", 
    F.count("transaction_id").over(window_spec))

# Merchant and category frequency
df_transactions = df_transactions.withColumn("merchant_count_per_card", 
    F.countDistinct("merchant").over(window_spec))
df_transactions = df_transactions.withColumn("category_count_per_card", 
    F.countDistinct("category").over(window_spec))

# Card's usual location
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

# Is weekend and night
df_transactions = df_transactions.withColumn("is_weekend", 
    F.when(F.col("day_of_week") >= 6, 1.0).otherwise(0.0))
df_transactions = df_transactions.withColumn("is_night", 
    F.when((F.col("hour_of_day") >= 22) | (F.col("hour_of_day") < 6), 1.0).otherwise(0.0))

# Category encoding
category_indexer = StringIndexer(inputCol="category", outputCol="category_index")
df_transactions = category_indexer.fit(df_transactions).transform(df_transactions)

print(f"Feature engineering complete")

# COMMAND ----------

# MAGIC %md ## Step 3: Create Tables

# COMMAND ----------

feature_table_name = f"{catalog}.{schema}.cctxn015310"
df_transactions.write.mode("overwrite").saveAsTable(feature_table_name)

training_table_name = f"{catalog}.{schema}.cctd015310"
df_transactions.write.mode("overwrite").saveAsTable(training_table_name)

print(f"Feature table: {feature_table_name}")
print(f"Training dataset: {training_table_name}")

# COMMAND ----------

# MAGIC %md ## Step 4: Train Classifier

# COMMAND ----------

spark_df = spark.table(feature_table_name)

feature_cols = [
    'amount', 'amount_log', 'amount_mean', 'amount_std', 'amount_zscore',
    'time_since_last_txn', 'txn_count_per_card',
    'merchant_count_per_card', 'category_count_per_card',
    'card_lat_mean', 'card_long_mean', 'distance_from_usual_km',
    'distance_std', 'distance_mean', 'distance_zscore',
    'hour_of_day', 'day_of_week', 'day_of_month', 'month',
    'is_weekend', 'is_night', 'category_index'
]

for c in feature_cols:
    spark_df = spark_df.withColumn(c, F.coalesce(F.col(c), F.lit(0.0)))

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
assembler_model = assembler.fit(spark_df)
df_assembled = assembler_model.transform(spark_df)

train_data, val_data = df_assembled.randomSplit([0.8, 0.2], seed=42)

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="is_fraud",
    numTrees=100,
    maxDepth=10,
    minInstancesPerNode=5,
    seed=42,
    subsamplingRate=0.8
)

rf_model = rf.fit(train_data)
val_predictions = rf_model.transform(val_data)

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

mlflow.set_experiment(f"/Users/benedict@hopsworks.ai/{prefix}/ccfraud_experiment")

with mlflow.start_run():
    mlflow.log_param("num_trees", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_metric("val_roc_auc", val_auc)
    mlflow.spark.log_model(rf_model, "model")
    
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
    mv = mlflow.register_model(
        model_uri=model_uri,
        name=f"{catalog}.{schema}.ccmodel015310"
    )
    print(f"Model registered: {mv.name}, version: {mv.version}")

# COMMAND ----------

# MAGIC %md ## Step 6: Score Transactions

# COMMAND ----------

# Apply same feature engineering to score data
df_score = df_score.withColumn("hour_of_day", F.hour("datetime"))
df_score = df_score.withColumn("day_of_week", F.dayofweek("datetime"))
df_score = df_score.withColumn("day_of_month", F.dayofmonth("datetime"))
df_score = df_score.withColumn("month", F.month("datetime"))
df_score = df_score.withColumn("amount_log", F.log1p("amount"))

card_stats = spark_df.select("cc_num", "amount_mean", "amount_std", "card_lat_mean", "card_long_mean", "merchant_count_per_card", "category_count_per_card", "distance_std", "distance_mean", "txn_count_per_card").distinct()
df_score = df_score.join(card_stats, on="cc_num", how="left")

for c in ["amount_mean", "amount_std", "card_lat_mean", "card_long_mean", "merchant_count_per_card", "category_count_per_card", "distance_std", "distance_mean", "txn_count_per_card"]:
    df_score = df_score.withColumn(c, F.coalesce(F.col(c), F.lit(0.0)))

df_score = df_score.withColumn("amount_zscore", 
    (F.col("amount") - F.col("amount_mean")) / (F.col("amount_std") + 1e-6))
df_score = df_score.withColumn("time_since_last_txn", F.lit(999.0))
df_score = df_score.withColumn("distance_from_usual_km", 
    haversine(F.col("lat"), F.col("long"), F.col("card_lat_mean"), F.col("card_long_mean")))
df_score = df_score.withColumn("distance_zscore", 
    (F.col("distance_from_usual_km") - F.col("distance_mean")) / (F.col("distance_std") + 1e-6))
df_score = df_score.withColumn("is_weekend", F.lit(0.0))
df_score = df_score.withColumn("is_night", F.lit(0.0))

category_indexer_model = category_indexer.fit(df_score)
df_score = category_indexer_model.transform(df_score)

for c in feature_cols:
    df_score = df_score.withColumn(c, F.coalesce(F.col(c), F.lit(0.0)))

df_score_assembled = assembler_model.transform(df_score)
score_predictions = rf_model.transform(df_score_assembled)

predictions_df = score_predictions.select(
    "transaction_id",
    F.col("probability")[1].alias("fraud_probability")
)

print(f"Predictions count: {predictions_df.count()}")

# COMMAND ----------

# MAGIC %md ## Step 7: Create Predictions Table

# COMMAND ----------

predictions_table_name = f"{catalog}.{schema}.ccpred015310"
predictions_df.write.mode("overwrite").saveAsTable(predictions_table_name)

print(f"Predictions table: {predictions_table_name}")

# COMMAND ----------

# Publish for online lookup
try:
    from databricks.feature_store import FeatureStoreClient
    fs = FeatureStoreClient()
    fs.publish_table(name=feature_table_name, online=True)
    fs.publish_table(name=predictions_table_name, online=True)
    print("Published tables for online lookup")
except Exception as e:
    print(f"Could not publish online: {e}")

print("\n=== PIPELINE COMPLETE ===")
print(f"Feature group: {catalog}.{schema}.cctxn015310")
print(f"Training dataset: {catalog}.{schema}.cctd015310")
print(f"Model: {catalog}.{schema}.ccmodel015310")
print(f"Predictions table: {catalog}.{schema}.ccpred015310")
print(f"Validation ROC AUC: {val_auc:.4f}")
