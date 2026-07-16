# Databricks notebook source
# MAGIC %md
# MAGIC # Credit Card Fraud Detection Pipeline

# COMMAND ----------

# Load data from local files
import os
print(f"Current directory: {os.getcwd()}")
print(f"Files: {os.listdir('.')}")

# Try to read from data directory
try:
    transactions_df = spark.read.csv("data/transactions.csv", header=True, inferSchema=True)
    score_df = spark.read.csv("data/score_transactions.csv", header=True, inferSchema=True)
    print(f"Loaded {transactions_df.count()} training transactions")
    print(f"Loaded {score_df.count()} score transactions")
except Exception as e:
    print(f"Error reading from data/: {e}")
    # Try from /Workspace
    try:
        transactions_df = spark.read.csv("/Workspace/data/transactions.csv", header=True, inferSchema=True)
        score_df = spark.read.csv("/Workspace/data/score_transactions.csv", header=True, inferSchema=True)
        print(f"Loaded from /Workspace/data/")
    except Exception as e2:
        print(f"Error reading from /Workspace/data/: {e2}")
        # Try from dbfs
        try:
            transactions_df = spark.read.csv("dbfs:/FileStore/data/transactions.csv", header=True, inferSchema=True)
            score_df = spark.read.csv("dbfs:/FileStore/data/score_transactions.csv", header=True, inferSchema=True)
            print(f"Loaded from dbfs:/FileStore/data/")
        except Exception as e3:
            print(f"Error reading from dbfs: {e3}")

# COMMAND ----------

# If we got here, we have the data. Now do feature engineering
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType
import math

# Convert datetime
transactions_df = transactions_df.withColumn("datetime", F.to_timestamp("datetime"))
score_df = score_df.withColumn("datetime", F.to_timestamp("datetime"))

# Time features
for df in [transactions_df, score_df]:
    df = df.withColumn("hour_of_day", F.hour("datetime"))
    df = df.withColumn("day_of_week", F.dayofweek("datetime") - 1)
    df = df.withColumn("amount_log", F.log(F.col("amount") + 1))

# Haversine UDF
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

haversine_udf = udf(haversine, FloatType())

# Card-level stats from training data
card_stats = transactions_df.groupBy("cc_num").agg(
    F.avg("amount").alias("card_avg_amount"),
    F.stddev("amount").alias("card_std_amount"),
    F.count("*").alias("card_txn_count"),
    F.avg("lat").alias("card_avg_lat"),
    F.avg("long").alias("card_avg_long")
)

merchant_freq = transactions_df.groupBy("cc_num", "merchant").agg(F.count("*").alias("mc"))
merchant_freq = merchant_freq.groupBy("cc_num").agg(
    F.max("mc").alias("max_merchant_freq"),
    F.countDistinct("merchant").alias("unique_merchants")
)

category_freq = transactions_df.groupBy("cc_num", "category").agg(F.count("*").alias("cc"))
category_freq = category_freq.groupBy("cc_num").agg(F.max("cc").alias("max_category_freq"))

# Join to both datasets
for df in [transactions_df, score_df]:
    df = df.join(card_stats, "cc_num", "left")
    df = df.join(merchant_freq, "cc_num", "left")
    df = df.join(category_freq, "cc_num", "left")
    df = df.withColumn("amount_dev", F.when(F.col("card_avg_amount") > 0, F.col("amount") / F.col("card_avg_amount")).otherwise(0))
    df = df.withColumn("geo_dist", haversine_udf(F.col("lat"), F.col("long"), F.col("card_avg_lat"), F.col("card_avg_long")))

# Fill nulls
fill_cols = ["card_avg_amount", "card_std_amount", "card_txn_count", "max_merchant_freq", 
            "unique_merchants", "max_category_freq", "card_avg_lat", "card_avg_long", 
            "geo_dist", "amount_dev"]
for df in [transactions_df, score_df]:
    for col in fill_cols:
        df = df.fillna({col: 0})
    df = df.replace(float("inf"), 0)

# Transaction velocity (24h)
transactions_df.createOrReplaceTempView("txn")
vel_df = spark.sql("SELECT t1.transaction_id, COUNT(t2.transaction_id)-1 as txn_24h FROM txn t1 LEFT JOIN txn t2 ON t1.cc_num=t2.cc_num AND t2.datetime>=t1.datetime-INTERVAL 24 HOURS AND t2.datetime<t1.datetime GROUP BY t1.transaction_id")
transactions_df = transactions_df.join(vel_df, "transaction_id", "left").fillna({"txn_24h": 0})

# Score velocity using combined data
combined = transactions_df.unionByName(score_df)
combined.createOrReplaceTempView("combined")
score_vel_df = spark.sql("SELECT t1.transaction_id, COUNT(t2.transaction_id)-1 as txn_24h FROM combined t1 LEFT JOIN combined t2 ON t1.cc_num=t2.cc_num AND t2.datetime>=t1.datetime-INTERVAL 24 HOURS AND t2.datetime<t1.datetime GROUP BY t1.transaction_id")
score_df = score_df.join(score_vel_df, "transaction_id", "left").fillna({"txn_24h": 0})

print("Features engineered")

# COMMAND ----------

# Create Feature Group
from databricks.feature_store import FeatureStoreClient
fs = FeatureStoreClient()

fg_df = transactions_df.select([
    "transaction_id", "cc_num", "datetime", "amount", "merchant", "category", "lat", "long", "is_fraud",
    "hour_of_day", "day_of_week", "amount_log", "txn_24h", "card_avg_amount", "card_std_amount", 
    "card_txn_count", "amount_dev", "max_merchant_freq", "unique_merchants", "max_category_freq", 
    "geo_dist", "card_avg_lat", "card_avg_long"
])

try:
    fs.create_table(name="workspace.mlpab958e4d.cctxn015310", primary_keys=["transaction_id"], df=fg_df)
    print("Created feature group: workspace.mlpab958e4d.cctxn015310")
except:
    fs.write_table(name="workspace.mlpab958e4d.cctxn015310", df=fg_df, mode="overwrite")
    print("Updated feature group: workspace.mlpab958e4d.cctxn015310")

# Create Training Dataset
td_df = transactions_df.select([
    "transaction_id", "amount", "hour_of_day", "day_of_week", "amount_log", "txn_24h",
    "card_avg_amount", "card_std_amount", "card_txn_count", "amount_dev", 
    "max_merchant_freq", "unique_merchants", "max_category_freq", "geo_dist", "lat", "long", "is_fraud"
])

try:
    fs.create_table(name="workspace.mlpab958e4d.cctd015310", primary_keys=["transaction_id"], df=td_df)
    print("Created training dataset: workspace.mlpab958e4d.cctd015310")
except:
    fs.write_table(name="workspace.mlpab958e4d.cctd015310", df=td_df, mode="overwrite")
    print("Updated training dataset: workspace.mlpab958e4d.cctd015310")

# COMMAND ----------

# Train Model
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import mlflow
from mlflow.models.signature import infer_signature

# Prepare features
cat_cols = ["merchant", "category"]
indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep").fit(td_df) for c in cat_cols]
encoder = OneHotEncoder(inputCols=[f"{c}_idx" for c in cat_cols], outputCols=[f"{c}_enc" for c in cat_cols])

num_cols = ["amount", "hour_of_day", "day_of_week", "amount_log", "txn_24h", "card_avg_amount", 
           "card_std_amount", "card_txn_count", "amount_dev", "max_merchant_freq", 
           "unique_merchants", "max_category_freq", "geo_dist", "lat", "long"]

assembler = VectorAssembler(inputCols=num_cols + [f"{c}_enc" for c in cat_cols], outputCol="features")
rf = RandomForestClassifier(labelCol="is_fraud", featuresCol="features", numTrees=200, maxDepth=6, seed=42)
pipeline = Pipeline(stages=[*indexers, encoder, assembler, rf])

train_data, test_data = td_df.randomSplit([0.8, 0.2], seed=42)
model = pipeline.fit(train_data)

predictions = model.transform(test_data)
evaluator = BinaryClassificationEvaluator(labelCol="is_fraud", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
auc = evaluator.evaluate(predictions)
print(f"Test AUC: {auc}")

# Register Model
mlflow.set_experiment("/Users/mlpab958e4d/fraud")
with mlflow.start_run():
    mlflow.log_metric("auc", auc)
    model_path = mlflow.spark.log_model(model, "model", signature=infer_signature(train_data, model), registered_model_name="workspace.mlpab958e4d.ccmodel015310")
    mlflow.register_model(model_path, "workspace.mlpab958e4d.ccmodel015310")
    print("Registered model: workspace.mlpab958e4d.ccmodel015310")

# COMMAND ----------

# Score Transactions
score_cols = ["amount", "hour_of_day", "day_of_week", "amount_log", "txn_24h", 
              "card_avg_amount", "card_std_amount", "card_txn_count", "amount_dev", 
              "max_merchant_freq", "unique_merchants", "max_category_freq", "geo_dist", 
              "lat", "long", "merchant", "category", "transaction_id"]

score_data = score_df.select(score_cols)
score_preds = model.transform(score_data)

get_prob = udf(lambda v: float(v[1]) if v and len(v) > 1 else 0.0, FloatType())
result_df = score_preds.select("transaction_id", get_prob("probability").alias("fraud_probability"))
result_df = result_df.withColumn("fraud_probability", F.when(F.col("fraud_probability") < 0, 0).when(F.col("fraud_probability") > 1, 1).otherwise(F.col("fraud_probability")))

print(f"Scored {result_df.count()} transactions")

# Create Predictions Table
try:
    fs.create_table(name="workspace.mlpab958e4d.ccpred015310", primary_keys=["transaction_id"], df=result_df)
    print("Created predictions: workspace.mlpab958e4d.ccpred015310")
except:
    fs.write_table(name="workspace.mlpab958e4d.ccpred015310", df=result_df, mode="overwrite")
    print("Updated predictions: workspace.mlpab958e4d.ccpred015310")

# Also save as regular table
result_df.write.saveAsTable("workspace.mlpab958e4d.ccpred015310", mode="overwrite")

# Create online table
try:
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()
    w.online_tables.create(name="mlpab958e4d_preds_online", source_table_name="workspace.mlpab958e4d.ccpred015310", comment="Fraud predictions online")
    print("Created online table")
except Exception as e:
    print(f"Online table error: {e}")

print(f"PIPELINE COMPLETE - AUC: {auc}")
