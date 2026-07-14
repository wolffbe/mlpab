#!/usr/bin/env python3
"""
Simple, direct pipeline execution.
"""

import os
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import compute

# Environment variables
SCHEMA = os.environ.get("MLPAB_DATABRICKS_SCHEMA", "workspace.mlpab958e4d")
PREFIX = os.environ.get("MLPAB_DATABRICKS_PREFIX", "mlpab958e4d")

# Object names
FG_NAME = "cctxn015310"
TD_NAME = "cctd015310" 
MODEL_NAME = "ccmodel015310"
PRED_NAME = "ccpred015310"

FG_FULL = f"{SCHEMA}.{FG_NAME}"
TD_FULL = f"{SCHEMA}.{TD_NAME}"
MODEL_FULL = f"{SCHEMA}.{MODEL_NAME}"
PRED_FULL = f"{SCHEMA}.{PRED_NAME}"

print(f"Schema: {SCHEMA}")
print(f"Prefix: {PREFIX}")

# Initialize workspace client
w = WorkspaceClient()
current_user = w.current_user.me().display_name

# Step 1: Upload data
print("\n=== Uploading data ===")
with open("data/transactions.csv", "rb") as f:
    w.dbfs.upload("dbfs:/FileStore/transactions.csv", f, overwrite=True)
print("✓ Uploaded transactions.csv")

with open("data/score_transactions.csv", "rb") as f:
    w.dbfs.upload("dbfs:/FileStore/score_transactions.csv", f, overwrite=True)
print("✓ Uploaded score_transactions.csv")

# Step 2: Create cluster
print("\n=== Creating cluster ===")
cluster_name = f"{PREFIX}_cluster"

try:
    clusters = w.clusters.list()
    cluster = next((c for c in clusters if c.cluster_name == cluster_name), None)
    
    if cluster:
        cluster_id = cluster.cluster_id
        print(f"✓ Using existing cluster: {cluster_name}")
    else:
        cluster = w.clusters.create(
            cluster_name=cluster_name,
            node_type_id="Standard_DS3_v2",
            spark_version="14.3.x-scala2.12",
            num_workers=2
        )
        cluster_id = cluster.cluster_id
        print(f"✓ Created cluster: {cluster_name}")
        
        # Wait for cluster
        while True:
            info = w.clusters.get(cluster_id=cluster_id)
            if info.state == "RUNNING":
                break
            elif info.state == "ERROR":
                print(f"✗ Cluster error: {info.state_message}")
                exit(1)
            time.sleep(10)
        print("✓ Cluster running")
        
except Exception as e:
    print(f"✗ Cluster error: {e}")
    exit(1)

# Step 3: Create and run notebook
print("\n=== Creating notebook ===")

# Create the pipeline notebook content
notebook_content = f'''# Fraud Detection Pipeline

# Load data
transactions_df = spark.read.csv("dbfs:/FileStore/transactions.csv", header=True, inferSchema=True)
score_df = spark.read.csv("dbfs:/FileStore/score_transactions.csv", header=True, inferSchema=True)

print(f"Loaded {{transactions_df.count()}} training transactions")
print(f"Loaded {{score_df.count()}} score transactions")

# Feature Engineering
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
    fs.create_table(name="{FG_FULL}", primary_keys=["transaction_id"], df=fg_df)
    print(f"Created feature group: {FG_FULL}")
except:
    fs.write_table(name="{FG_FULL}", df=fg_df, mode="overwrite")
    print(f"Updated feature group: {FG_FULL}")

# Create Training Dataset
td_df = transactions_df.select([
    "transaction_id", "amount", "hour_of_day", "day_of_week", "amount_log", "txn_24h",
    "card_avg_amount", "card_std_amount", "card_txn_count", "amount_dev", 
    "max_merchant_freq", "unique_merchants", "max_category_freq", "geo_dist", "lat", "long", "is_fraud"
])

try:
    fs.create_table(name="{TD_FULL}", primary_keys=["transaction_id"], df=td_df)
    print(f"Created training dataset: {TD_FULL}")
except:
    fs.write_table(name="{TD_FULL}", df=td_df, mode="overwrite")
    print(f"Updated training dataset: {TD_FULL}")

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
mlflow.set_experiment(f"/Users/{{current_user}}/{PREFIX}/fraud")
with mlflow.start_run():
    mlflow.log_metric("auc", auc)
    model_path = mlflow.spark.log_model(model, "model", signature=infer_signature(train_data, model), registered_model_name="{MODEL_FULL}")
    mlflow.register_model(model_path, "{MODEL_FULL}")
    print(f"Registered model: {MODEL_FULL}")

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
    fs.create_table(name="{PRED_FULL}", primary_keys=["transaction_id"], df=result_df)
    print(f"Created predictions: {PRED_FULL}")
except:
    fs.write_table(name="{PRED_FULL}", df=result_df, mode="overwrite")
    print(f"Updated predictions: {PRED_FULL}")

# Also save as regular table
result_df.write.saveAsTable("{PRED_FULL}", mode="overwrite")

# Create online table
try:
    w.online_tables.create(name="{PREFIX}_preds_online", source_table_name="{PRED_FULL}", comment="Fraud predictions online")
    print("Created online table")
except Exception as e:
    print(f"Online table error: {{e}}")

print(f"PIPELINE COMPLETE - AUC: {auc}")
'''

notebook_path = f"/Users/{current_user}/{PREFIX}/fraud_pipeline"
w.workspace.upload(path=notebook_path, content=notebook_content, language="PYTHON", overwrite=True)
print(f"✓ Created notebook: {notebook_path}")

# Step 4: Run as job
print("\n=== Running job ===")
job_name = f"{PREFIX}_fraud_job"

try:
    job = w.jobs.create(
        name=job_name,
        tasks=[{
            "task_key": "run_pipeline",
            "notebook_task": {"notebook_path": notebook_path},
            "existing_cluster_id": cluster_id
        }]
    )
    print(f"✓ Created job: {job.job_id}")
    
    # Run job
    run = w.jobs.run_now(job_id=job.job_id)
    print(f"✓ Started run: {run.run_id}")
    
    # Wait for completion
    max_wait = 3600
    start = time.time()
    
    while time.time() - start < max_wait:
        time.sleep(30)
        run_info = w.jobs.get_run(run_id=run.run_id)
        state = run_info.state.life_cycle_state
        result = getattr(run_info.state, 'result_state', None)
        
        if state == "TERMINATED":
            if result == "SUCCESS":
                print("✓ Job succeeded!")
                break
            else:
                print(f"✗ Job failed: {result}")
                if hasattr(run_info.state, 'state_message'):
                    print(f"Error: {run_info.state.state_message}")
                break
        elif state in ["SKIPPED", "INTERNAL_ERROR"]:
            print(f"✗ Job {state}")
            break
        else:
            print(f"  Status: {state}")
    
    if time.time() - start >= max_wait:
        print("✗ Job timed out")

except Exception as e:
    print(f"✗ Job error: {e}")

print("\n=== Done ===")
