# Databricks notebook source
# MAGIC %md
# MAGIC # Credit Card Fraud Detection Pipeline - Debug Version

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

# MAGIC %md ## Step 1: Load Data from Volume

# COMMAND ----------

# Read the CSV files from the volume
volume_path = "/Volumes/workspace/mlpabfcf9c1/ccfraud_data/"
transactions_path = f"{volume_path}transactions.csv"
score_path = f"{volume_path}score_transactions.csv"

print(f"Reading transactions from: {transactions_path}")

try:
    df_transactions = spark.read.csv(transactions_path, header=True, inferSchema=True)
    print(f"Read transactions: {df_transactions.count()} rows")
except Exception as e:
    print(f"Error reading transactions: {e}")
    import traceback
    traceback.print_exc()
    raise

try:
    df_score = spark.read.csv(score_path, header=True, inferSchema=True)
    print(f"Read score transactions: {df_score.count()} rows")
except Exception as e:
    print(f"Error reading score transactions: {e}")
    import traceback
    traceback.print_exc()
    raise

# Convert datetime string to timestamp
df_transactions = df_transactions.withColumn("datetime", F.to_timestamp("datetime", "yyyy-MM-dd'T'HH:mm:ss'Z'"))
df_score = df_score.withColumn("datetime", F.to_timestamp("datetime", "yyyy-MM-dd'T'HH:mm:ss'Z'"))

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
try:
    df_transactions.write.mode("overwrite").saveAsTable(feature_table_name)
    print(f"Feature table created: {feature_table_name}")
except Exception as e:
    print(f"Error creating feature table: {e}")
    import traceback
    traceback.print_exc()
    raise

training_table_name = f"{catalog}.{schema}.cctd015310"
try:
    df_transactions.write.mode("overwrite").saveAsTable(training_table_name)
    print(f"Training dataset created: {training_table_name}")
except Exception as e:
    print(f"Error creating training table: {e}")
    import traceback
    traceback.print_exc()
    raise

# COMMAND ----------

print("Steps 1-3 completed successfully")
