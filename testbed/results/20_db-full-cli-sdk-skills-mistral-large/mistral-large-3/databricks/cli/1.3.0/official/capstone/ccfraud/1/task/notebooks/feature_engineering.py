# Databricks notebook source
# MAGIC %md
# Feature Engineering for Fraud Detection

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql.functions import col, unix_timestamp, lag, count, sum, mean, stddev, abs, sqrt, pow, lit, when
from pyspark.sql.window import Window
import mlflow

# Initialize Feature Engineering Client
fe = FeatureEngineeringClient()

# Read raw data
raw_data_path = f"/Volumes/workspace/{dbutils.widgets.get('schema')}/raw/transactions.csv"
transactions_df = spark.read.csv(raw_data_path, header=True, inferSchema=True)

# COMMAND ----------

# Feature Engineering: Transaction velocity, geo distance, amount signals
# 1. Transaction velocity per card (count of transactions in the last 1 hour)
# 2. Average amount per card in the last 1 hour
# 3. Standard deviation of amount per card in the last 1 hour
# 4. Geo distance from the previous transaction

# Convert datetime to timestamp
transactions_df = transactions_df.withColumn("timestamp", unix_timestamp(col("datetime")))

# Window for time-based features (1 hour)
windowSpec = Window.partitionBy("cc_num").orderBy("timestamp").rangeBetween(-3600, 0)

# Feature 1: Transaction velocity (count in last 1 hour)
transactions_df = transactions_df.withColumn("txn_velocity_1h", count("*").over(windowSpec))

# Feature 2: Average amount in last 1 hour
transactions_df = transactions_df.withColumn("avg_amount_1h", mean("amount").over(windowSpec))

# Feature 3: Standard deviation of amount in last 1 hour
transactions_df = transactions_df.withColumn("std_amount_1h", stddev("amount").over(windowSpec))

# Feature 4: Geo distance from previous transaction
windowSpecPrev = Window.partitionBy("cc_num").orderBy("timestamp")
transactions_df = transactions_df.withColumn("prev_lat", lag("lat").over(windowSpecPrev))
transactions_df = transactions_df.withColumn("prev_long", lag("long").over(windowSpecPrev))
transactions_df = transactions_df.withColumn(
    "geo_distance",
    when(
        (col("prev_lat").isNotNull()) & (col("prev_long").isNotNull()),
        sqrt(pow(abs(col("lat") - col("prev_lat")) * 111.32, 2) + pow(abs(col("long") - col("prev_long")) * 111.32 * cos(radians(col("lat"))), 2))
    ).otherwise(lit(0.0))
)

# COMMAND ----------

# Create Feature Table
feature_table_name = f"workspace.{dbutils.widgets.get('schema')}.cctxne0b071"

# Write to Feature Store
fe.create_table(
    name=feature_table_name,
    primary_keys=["transaction_id"],
    df=transactions_df,
    partition_columns=[],
    description="Fraud detection features: transaction velocity, geo distance, amount statistics"
)

# COMMAND ----------

# Log the feature table in MLflow
with mlflow.start_run():
    mlflow.log_param("feature_table_name", feature_table_name)
    mlflow.log_param("primary_keys", ["transaction_id"])
    mlflow.log_param("features", ["txn_velocity_1h", "avg_amount_1h", "std_amount_1h", "geo_distance"])