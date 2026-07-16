# Databricks notebook source
# MAGIC %md
# Full Fraud Detection Pipeline

# COMMAND ----------

from pyspark.sql.functions import col, unix_timestamp, lag, count, mean, stddev, abs, sqrt, pow, lit, when
from pyspark.sql.window import Window
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import mlflow
import pandas as pd

# COMMAND ----------

# Read raw data
schema = dbutils.widgets.get('schema')
raw_data_path = f"/Volumes/workspace/{schema}/raw/transactions.csv"
transactions_df = spark.read.csv(raw_data_path, header=True, inferSchema=True)

# COMMAND ----------

# Feature Engineering: Transaction velocity, geo distance, amount signals
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

# Write feature table to Unity Catalog
feature_table_name = f"workspace.{schema}.cctxne0b071"
transactions_df.write.saveAsTable(feature_table_name)

# COMMAND ----------

# Split data into train and test
train_df = transactions_df.filter(col("datetime") < "2025-12-01")
pandas_train = train_df.toPandas()

X_train = pandas_train[["txn_velocity_1h", "avg_amount_1h", "std_amount_1h", "geo_distance"]]
y_train = pandas_train["is_fraud"]

# COMMAND ----------

# Train model
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Log model and parameters
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 100)
    
    # Evaluate on training data (for demonstration)
    y_pred_proba = model.predict_proba(X_train)[:, 1]
    auc = roc_auc_score(y_train, y_pred_proba)
    mlflow.log_metric("train_roc_auc", auc)
    
    # Register model
    model_name = f"workspace.{schema}.ccmodele0b071"
    mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", model_name)

# COMMAND ----------

# Read scoring data
raw_score_path = f"/Volumes/workspace/{schema}/raw/score_transactions.csv"
score_df = spark.read.csv(raw_score_path, header=True, inferSchema=True)

# Feature Engineering for scoring data
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

# Score data
pandas_score = score_df.toPandas()
X_score = pandas_score[["txn_velocity_1h", "avg_amount_1h", "std_amount_1h", "geo_distance"]]

fraud_probability = model.predict_proba(X_score)[:, 1]
pandas_score["fraud_probability"] = fraud_probability

# COMMAND ----------

# Write predictions to Unity Catalog
predictions_df = spark.createDataFrame(pandas_score[["transaction_id", "fraud_probability"]])
predictions_table_name = f"workspace.{schema}.ccprede0b071"
predictions_df.write.saveAsTable(predictions_table_name)

# COMMAND ----------

# Log feature table and model in MLflow
with mlflow.start_run():
    mlflow.log_param("feature_table_name", feature_table_name)
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("predictions_table_name", predictions_table_name)