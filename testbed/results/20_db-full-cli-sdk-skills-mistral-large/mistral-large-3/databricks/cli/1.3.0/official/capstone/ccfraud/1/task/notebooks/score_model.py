# Databricks notebook source
# MAGIC %md
# Score Fraud Detection Model

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql.functions import col, lit
import mlflow
import pandas as pd

# Initialize Feature Engineering Client
fe = FeatureEngineeringClient()

# COMMAND ----------

# Read feature table and model
feature_table_name = f"workspace.{dbutils.widgets.get('schema')}.cctxne0b071"
model_name = f"workspace.{dbutils.widgets.get('schema')}.ccmodele0b071"

# Read scoring data
raw_data_path = f"/Volumes/workspace/{dbutils.widgets.get('schema')}/raw/score_transactions.csv"
score_df = spark.read.csv(raw_data_path, header=True, inferSchema=True)

# COMMAND ----------

# Feature Engineering for scoring data (reuse logic from feature engineering)
from pyspark.sql.functions import unix_timestamp, lag, count, mean, stddev, abs, sqrt, pow, when
from pyspark.sql.window import Window

# Convert datetime to timestamp
score_df = score_df.withColumn("timestamp", unix_timestamp(col("datetime")))

# Window for time-based features (1 hour)
windowSpec = Window.partitionBy("cc_num").orderBy("timestamp").rangeBetween(-3600, 0)

# Feature 1: Transaction velocity (count in last 1 hour)
score_df = score_df.withColumn("txn_velocity_1h", count("*").over(windowSpec))

# Feature 2: Average amount in last 1 hour
score_df = score_df.withColumn("avg_amount_1h", mean("amount").over(windowSpec))

# Feature 3: Standard deviation of amount in last 1 hour
score_df = score_df.withColumn("std_amount_1h", stddev("amount").over(windowSpec))

# Feature 4: Geo distance from previous transaction
windowSpecPrev = Window.partitionBy("cc_num").orderBy("timestamp")
score_df = score_df.withColumn("prev_lat", lag("lat").over(windowSpecPrev))
score_df = score_df.withColumn("prev_long", lag("long").over(windowSpecPrev))
score_df = score_df.withColumn(
    "geo_distance",
    when(
        (col("prev_lat").isNotNull()) & (col("prev_long").isNotNull()),
        sqrt(pow(abs(col("lat") - col("prev_lat")) * 111.32, 2) + pow(abs(col("long") - col("prev_long")) * 111.32 * cos(radians(col("lat"))), 2))
    ).otherwise(lit(0.0))
)

# COMMAND ----------

# Load model
model_uri = f"models:/{model_name}/latest"
model = mlflow.sklearn.load_model(model_uri)

# COMMAND ----------

# Score data
pandas_score = score_df.toPandas()
X_score = pandas_score[["txn_velocity_1h", "avg_amount_1h", "std_amount_1h", "geo_distance"]]

fraud_probability = model.predict_proba(X_score)[:, 1]
pandas_score["fraud_probability"] = fraud_probability

# COMMAND ----------

# Write predictions to feature table
predictions_df = spark.createDataFrame(pandas_score[["transaction_id", "fraud_probability"]])
predictions_table_name = f"workspace.{dbutils.widgets.get('schema')}.ccprede0b071"

fe.create_table(
    name=predictions_table_name,
    primary_keys=["transaction_id"],
    df=predictions_df,
    description="Fraud probability predictions for transactions"
)

# COMMAND ----------

# Enable online lookup
fe.publish_table(predictions_table_name)