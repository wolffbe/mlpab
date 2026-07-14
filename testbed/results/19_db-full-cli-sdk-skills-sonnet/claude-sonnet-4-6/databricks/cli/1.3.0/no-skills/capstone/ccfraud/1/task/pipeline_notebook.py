# Databricks notebook source
# MAGIC %md # Credit Card Fraud Detection Pipeline

# COMMAND ----------
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import *
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

print("Imports successful")

# COMMAND ----------
CATALOG = "workspace"
SCHEMA = "mlpab3f803c"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/data_volume"
FG_TABLE = f"{CATALOG}.{SCHEMA}.cctxn739ee9"
TD_TABLE = f"{CATALOG}.{SCHEMA}.cctd739ee9"
PRED_TABLE = f"{CATALOG}.{SCHEMA}.ccpred739ee9"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.ccmodel739ee9"
EXPERIMENT_PATH = f"/Users/benedict@logicalclocks.com/mlpab3f803c/ccfraud_experiment"

# COMMAND ----------
# Read training data into pandas for feature engineering
txn_df = spark.read.option("header", True).option("inferSchema", True).csv(f"{VOLUME_PATH}/transactions.csv")
txn_df = txn_df.withColumn("ts", F.to_timestamp("datetime"))
txn_df = txn_df.withColumn("ts_unix", F.unix_timestamp("ts"))
print(f"Training rows: {txn_df.count()}")

# COMMAND ----------
# Feature engineering using Spark window functions (supported in serverless)
# Use ROWS BETWEEN (not RANGE) to avoid timestamp/bigint mismatch
window_card = Window.partitionBy("cc_num").orderBy("ts_unix")
window_card_24h = window_card.rowsBetween(-100, 0)  # Approximate 24h window by rows
window_card_prior = window_card.rowsBetween(Window.unboundedPreceding, -1)

# Time features
txn_df = txn_df.withColumn("hour_of_day", F.hour("ts"))
txn_df = txn_df.withColumn("day_of_week", F.dayofweek("ts"))
txn_df = txn_df.withColumn("is_weekend", (F.col("day_of_week").isin([1, 7])).cast(IntegerType()))
txn_df = txn_df.withColumn("is_night", ((F.col("hour_of_day") >= 22) | (F.col("hour_of_day") <= 5)).cast(IntegerType()))

# Rolling features per card (row-based windows)
txn_df = txn_df.withColumn("txn_count_24h", F.count("transaction_id").over(window_card_24h))
txn_df = txn_df.withColumn("amount_sum_24h", F.sum("amount").over(window_card_24h))

# Historical stats per card
txn_df = txn_df.withColumn("mean_amount_hist", F.mean("amount").over(window_card_prior))
txn_df = txn_df.withColumn("mean_amount_hist",
    F.when(F.col("mean_amount_hist").isNull(), F.col("amount")).otherwise(F.col("mean_amount_hist")))
txn_df = txn_df.withColumn("std_amount_hist", F.stddev("amount").over(window_card_prior))
txn_df = txn_df.withColumn("std_amount_hist",
    F.when(F.col("std_amount_hist").isNull(), 0.0).otherwise(F.col("std_amount_hist")))

# Geo features
txn_df = txn_df.withColumn("mean_lat", F.mean("lat").over(window_card_prior))
txn_df = txn_df.withColumn("mean_long", F.mean("long").over(window_card_prior))
txn_df = txn_df.withColumn("mean_lat",
    F.when(F.col("mean_lat").isNull(), F.col("lat")).otherwise(F.col("mean_lat")))
txn_df = txn_df.withColumn("mean_long",
    F.when(F.col("mean_long").isNull(), F.col("long")).otherwise(F.col("mean_long")))

# Derived features
txn_df = txn_df.withColumn("amount_deviation", F.col("amount") - F.col("mean_amount_hist"))
txn_df = txn_df.withColumn("amount_z_score",
    F.when(F.col("std_amount_hist") > 0,
           (F.col("amount") - F.col("mean_amount_hist")) / F.col("std_amount_hist")
    ).otherwise(0.0))
txn_df = txn_df.withColumn("geo_distance",
    F.sqrt(F.pow(F.col("lat") - F.col("mean_lat"), 2) + F.pow(F.col("long") - F.col("mean_long"), 2)))

# Category indicators
txn_df = txn_df.withColumn("is_online", (F.col("category") == "online").cast(IntegerType()))
txn_df = txn_df.withColumn("is_travel", (F.col("category") == "travel").cast(IntegerType()))
txn_df = txn_df.withColumn("is_cash_advance", (F.col("category") == "cash_advance").cast(IntegerType()))
txn_df = txn_df.withColumn("is_entertainment", (F.col("category") == "entertainment").cast(IntegerType()))

# Amount features
txn_df = txn_df.withColumn("log_amount", F.log1p("amount"))
txn_df = txn_df.withColumn("is_high_amount", (F.col("amount") > 500).cast(IntegerType()))
txn_df = txn_df.withColumn("is_round_amount", ((F.col("amount") % 10).cast(IntegerType()) == 0).cast(IntegerType()))

print("Features computed")

# COMMAND ----------
# Feature column list
feature_cols = [
    "amount", "log_amount", "hour_of_day", "day_of_week",
    "is_weekend", "is_night",
    "txn_count_24h", "amount_sum_24h",
    "mean_amount_hist", "std_amount_hist", "amount_deviation", "amount_z_score",
    "geo_distance",
    "is_online", "is_travel", "is_cash_advance", "is_entertainment",
    "is_high_amount", "is_round_amount"
]

# Select and save feature group (include lat/long for card stats computation during scoring)
feature_df = txn_df.select(
    ["transaction_id", "cc_num", "ts", "lat", "long"] + feature_cols + ["is_fraud"]
)
feature_df = feature_df.fillna(0)

feature_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(FG_TABLE)
print(f"Feature group written: {FG_TABLE} ({feature_df.count()} rows)")

# COMMAND ----------
# Write training dataset
training_df = spark.table(FG_TABLE)
training_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(TD_TABLE)
print(f"Training dataset written: {TD_TABLE}")

# COMMAND ----------
# Convert to pandas for ML training
pd_df = spark.table(TD_TABLE).select(feature_cols + ["is_fraud"]).fillna(0).toPandas()
print(f"Pandas dataframe: {pd_df.shape}")
print(f"Fraud rate: {pd_df['is_fraud'].mean():.3f}")

X = pd_df[feature_cols].values
y = pd_df["is_fraud"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# COMMAND ----------
mlflow.set_experiment(EXPERIMENT_PATH)

with mlflow.start_run() as run:
    # Train GBT classifier
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"Test AUC: {auc:.4f}")

    mlflow.log_metric("roc_auc", auc)
    mlflow.log_param("model_type", "GradientBoostingClassifier")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("learning_rate", 0.05)
    mlflow.log_param("feature_cols", str(feature_cols))

    # Register model in Unity Catalog
    input_example = pd.DataFrame(X_train[:3], columns=feature_cols)
    mlflow.sklearn.log_model(
        model,
        "model",
        input_example=input_example,
        registered_model_name=MODEL_NAME
    )

    run_id = run.info.run_id
    print(f"Run ID: {run_id}, AUC: {auc:.4f}")

# COMMAND ----------
# Score the scoring dataset
score_raw = spark.read.option("header", True).option("inferSchema", True).csv(f"{VOLUME_PATH}/score_transactions.csv")
score_raw = score_raw.withColumn("ts", F.to_timestamp("datetime"))
score_raw = score_raw.withColumn("ts_unix", F.unix_timestamp("ts"))

# Time features
score_raw = score_raw.withColumn("hour_of_day", F.hour("ts"))
score_raw = score_raw.withColumn("day_of_week", F.dayofweek("ts"))
score_raw = score_raw.withColumn("is_weekend", (F.col("day_of_week").isin([1, 7])).cast(IntegerType()))
score_raw = score_raw.withColumn("is_night", ((F.col("hour_of_day") >= 22) | (F.col("hour_of_day") <= 5)).cast(IntegerType()))
score_raw = score_raw.withColumn("log_amount", F.log1p("amount"))
score_raw = score_raw.withColumn("is_online", (F.col("category") == "online").cast(IntegerType()))
score_raw = score_raw.withColumn("is_travel", (F.col("category") == "travel").cast(IntegerType()))
score_raw = score_raw.withColumn("is_cash_advance", (F.col("category") == "cash_advance").cast(IntegerType()))
score_raw = score_raw.withColumn("is_entertainment", (F.col("category") == "entertainment").cast(IntegerType()))
score_raw = score_raw.withColumn("is_high_amount", (F.col("amount") > 500).cast(IntegerType()))
score_raw = score_raw.withColumn("is_round_amount", ((F.col("amount") % 10).cast(IntegerType()) == 0).cast(IntegerType()))

# Card stats from feature group
card_stats = spark.table(FG_TABLE).groupBy("cc_num").agg(
    F.mean("amount").alias("mean_amount_hist"),
    F.stddev("amount").alias("std_amount_hist"),
    F.mean("lat").alias("mean_lat"),
    F.mean("long").alias("mean_long"),
    F.mean("txn_count_24h").alias("txn_count_24h"),
    F.mean("amount_sum_24h").alias("amount_sum_24h")
)

# Global defaults
agg_row = spark.table(FG_TABLE).agg(
    F.mean("amount").alias("m_amt"),
    F.stddev("amount").alias("s_amt"),
    F.mean("txn_count_24h").alias("m_cnt"),
    F.mean("amount_sum_24h").alias("m_sum")
).collect()[0]
global_mean_amount = float(agg_row["m_amt"])
global_std_amount = float(agg_row["s_amt"])
global_mean_txn = float(agg_row["m_cnt"])
global_mean_sum = float(agg_row["m_sum"])

score_df = score_raw.join(card_stats, on="cc_num", how="left")
score_df = score_df.fillna({
    "mean_amount_hist": global_mean_amount,
    "std_amount_hist": global_std_amount,
    "mean_lat": 39.5,
    "mean_long": -98.35,
    "txn_count_24h": global_mean_txn,
    "amount_sum_24h": global_mean_sum
})

score_df = score_df.withColumn("amount_deviation", F.col("amount") - F.col("mean_amount_hist"))
score_df = score_df.withColumn("amount_z_score",
    F.when(F.col("std_amount_hist") > 0,
           (F.col("amount") - F.col("mean_amount_hist")) / F.col("std_amount_hist")
    ).otherwise(0.0))
score_df = score_df.withColumn("geo_distance",
    F.sqrt(F.pow(F.col("lat") - F.col("mean_lat"), 2) + F.pow(F.col("long") - F.col("mean_long"), 2)))

score_pd = score_df.select(["transaction_id"] + feature_cols).fillna(0).toPandas()
print(f"Score rows: {score_pd.shape}")

# COMMAND ----------
# Generate predictions
X_score = score_pd[feature_cols].values
y_score = model.predict_proba(X_score)[:, 1]

result_pd = pd.DataFrame({
    "transaction_id": score_pd["transaction_id"].values,
    "fraud_probability": y_score.tolist()
})
print(f"Predictions: {result_pd.shape}")
print(result_pd["fraud_probability"].describe())

# COMMAND ----------
# Write predictions to Delta table
result_spark = spark.createDataFrame(result_pd)
result_spark.write.format("delta").mode("overwrite").option("delta.enableChangeDataFeed", "true").saveAsTable(PRED_TABLE)
print(f"Predictions written: {PRED_TABLE}")

# Enable CDF
spark.sql(f"ALTER TABLE {PRED_TABLE} SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')")
print("CDF enabled")

# COMMAND ----------
# Create online table for low-latency lookup
import requests
import os

# Get host from Spark config (available in serverless env) or env var
try:
    workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
except:
    workspace_url = ""

host = os.environ.get("DATABRICKS_HOST", workspace_url)
token = os.environ.get("DATABRICKS_TOKEN", "")

# Try to get token from dbutils if env var not set
if not token:
    try:
        token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
    except:
        pass

# Try to get host from context if not set
if not host:
    try:
        host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().getOrElse(None)
    except:
        pass

print(f"Using host: {host}")
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
url = f"https://{host}/api/2.0/online-tables/tables"

online_table_spec = {
    "name": f"{CATALOG}.{SCHEMA}.ccpred739ee9_online",
    "spec": {
        "source_table_full_name": PRED_TABLE,
        "primary_key_columns": ["transaction_id"],
        "run_triggered": {
            "triggered_enable_continuous_auto_refresh": True
        }
    }
}

import json
response = requests.post(url, headers=headers, json=online_table_spec)
print(f"Online table status: {response.status_code}")
if response.status_code not in [200, 201]:
    print(f"Response: {response.text[:500]}")
else:
    print("Online table created successfully")

# COMMAND ----------
print("=== Pipeline Complete ===")
print(f"Feature group: {FG_TABLE}")
print(f"Training dataset: {TD_TABLE}")
print(f"Model: {MODEL_NAME}")
print(f"Predictions: {PRED_TABLE}")
print(f"Test AUC was: {auc:.4f}")
