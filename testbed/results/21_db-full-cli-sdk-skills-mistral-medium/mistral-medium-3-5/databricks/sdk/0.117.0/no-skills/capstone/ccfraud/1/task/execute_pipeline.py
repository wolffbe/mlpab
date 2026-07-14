#!/usr/bin/env python3
"""
Execute the fraud detection pipeline directly using Databricks command execution.
"""

import os
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import compute

# Environment variables
SCHEMA = os.environ.get("MLPAB_DATABRICKS_SCHEMA", "workspace.mlpab958e4d")
PREFIX = os.environ.get("MLPAB_DATABRICKS_PREFIX", "mlpab958e4d")

# Object names
FEATURE_GROUP_NAME = "cctxn015310"
TRAINING_DATASET_NAME = "cctd015310" 
MODEL_NAME = "ccmodel015310"
PREDICTIONS_TABLE_NAME = "ccpred015310"

# Full names
FG_FULL_NAME = f"{SCHEMA}.{FEATURE_GROUP_NAME}"
TD_FULL_NAME = f"{SCHEMA}.{TRAINING_DATASET_NAME}"
MODEL_FULL_NAME = f"{SCHEMA}.{MODEL_NAME}"
PRED_FULL_NAME = f"{SCHEMA}.{PREDICTIONS_TABLE_NAME}"

print(f"Schema: {SCHEMA}")
print(f"Feature Group: {FG_FULL_NAME}")
print(f"Training Dataset: {TD_FULL_NAME}")
print(f"Model: {MODEL_FULL_NAME}")
print(f"Predictions: {PRED_FULL_NAME}")

# Initialize workspace client
w = WorkspaceClient()

# Get current user
current_user = w.current_user.me().display_name
print(f"Current user: {current_user}")

# Step 1: Upload data files to DBFS
print("\n=== Step 1: Uploading data files ===")
try:
    with open("data/transactions.csv", "rb") as f:
        w.dbfs.upload("dbfs:/FileStore/transactions.csv", f, overwrite=True)
    print("✓ Uploaded transactions.csv")
    
    with open("data/score_transactions.csv", "rb") as f:
        w.dbfs.upload("dbfs:/FileStore/score_transactions.csv", f, overwrite=True)
    print("✓ Uploaded score_transactions.csv")
except Exception as e:
    print(f"✗ Error uploading files: {e}")
    exit(1)

# Step 2: Create a cluster
print("\n=== Step 2: Creating cluster ===")
cluster_name = f"{PREFIX}_cluster"

try:
    # Check if cluster exists
    clusters = w.clusters.list()
    cluster_exists = any(c.cluster_name == cluster_name for c in clusters)
    
    if cluster_exists:
        print(f"✓ Cluster {cluster_name} already exists")
        # Get the cluster ID
        cluster = next(c for c in clusters if c.cluster_name == cluster_name)
        cluster_id = cluster.cluster_id
    else:
        # Create a new cluster
        cluster = w.clusters.create(
            cluster_name=cluster_name,
            node_type_id="Standard_DS3_v2",
            spark_version="14.3.x-scala2.12",
            num_workers=2,
            autoscale=compute.ClusterAutoscale(min_workers=2, max_workers=4)
        )
        cluster_id = cluster.cluster_id
        print(f"✓ Created cluster: {cluster_name} (ID: {cluster_id})")
        
        # Wait for cluster to be ready
        print("Waiting for cluster to start...")
        cluster_info = w.clusters.get(cluster_id=cluster_id)
        while cluster_info.state != "RUNNING":
            time.sleep(10)
            cluster_info = w.clusters.get(cluster_id=cluster_id)
            print(f"  Cluster state: {cluster_info.state}")
        print("✓ Cluster is running")
        
except Exception as e:
    print(f"✗ Error with cluster: {e}")
    exit(1)

# Step 3: Create and run the pipeline using command execution
print("\n=== Step 3: Running pipeline ===")

# Create the pipeline script
pipeline_script = f"""
# Credit Card Fraud Detection Pipeline
import os
import math
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType
from databricks.feature_store import FeatureStoreClient
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import mlflow
from mlflow.models.signature import infer_signature

print("Loading data...")
transactions_df = spark.read.csv("dbfs:/FileStore/transactions.csv", header=True, inferSchema=True)
score_df = spark.read.csv("dbfs:/FileStore/score_transactions.csv", header=True, inferSchema=True)

print(f"Transactions: {{transactions_df.count()}}")
print(f"Score transactions: {{score_df.count()}}")

# Feature Engineering
print("Feature engineering...")

# Convert datetime
transactions_df = transactions_df.withColumn("datetime", F.to_timestamp("datetime"))
score_df = score_df.withColumn("datetime", F.to_timestamp("datetime"))

# Basic time features
for df in [transactions_df, score_df]:
    df = df.withColumn("hour_of_day", F.hour("datetime"))
    df = df.withColumn("day_of_week", F.dayofweek("datetime") - 1)
    df = df.withColumn("amount_log", F.log(F.col("amount") + 1))

# Haversine distance UDF
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

haversine_udf = udf(haversine, FloatType())

# Card-level statistics from training data
print("Computing card-level statistics...")
card_stats = transactions_df.groupBy("cc_num").agg(
    F.avg("amount").alias("card_avg_amount"),
    F.stddev("amount").alias("card_std_amount"),
    F.count("*").alias("card_txn_count"),
    F.avg("lat").alias("card_avg_lat"),
    F.avg("long").alias("card_avg_long")
)

merchant_freq = transactions_df.groupBy("cc_num", "merchant").agg(F.count("*").alias("merchant_count"))
merchant_freq = merchant_freq.groupBy("cc_num").agg(
    F.max("merchant_count").alias("max_merchant_freq"),
    F.countDistinct("merchant").alias("unique_merchants")
)

category_freq = transactions_df.groupBy("cc_num", "category").agg(F.count("*").alias("category_count"))
category_freq = category_freq.groupBy("cc_num").agg(F.max("category_count").alias("max_category_freq"))

# Join card-level features
for df in [transactions_df, score_df]:
    df = df.join(card_stats, "cc_num", "left")
    df = df.join(merchant_freq, "cc_num", "left")
    df = df.join(category_freq, "cc_num", "left")

# Compute derived features
for df in [transactions_df, score_df]:
    df = df.withColumn("amount_dev_from_avg", F.when(F.col("card_avg_amount") > 0, F.col("amount") / F.col("card_avg_amount")).otherwise(0))
    df = df.withColumn("geo_distance_from_avg", haversine_udf(F.col("lat"), F.col("long"), F.col("card_avg_lat"), F.col("card_avg_long")))

# Fill nulls
fill_cols = ["card_avg_amount", "card_std_amount", "card_txn_count", "max_merchant_freq", 
            "unique_merchants", "max_category_freq", "card_avg_lat", "card_avg_long", 
            "geo_distance_from_avg", "amount_dev_from_avg"]

for df in [transactions_df, score_df]:
    for col_name in fill_cols:
        df = df.fillna({col_name: 0})
    df = df.replace(float('inf'), 0)

# Transaction velocity (24h count)
print("Computing transaction velocity...")
transactions_df.createOrReplaceTempView("txn_data")
velocity_df = spark.sql("""
    SELECT t1.transaction_id, COUNT(t2.transaction_id) - 1 as txn_count_24h
    FROM txn_data t1
    LEFT JOIN txn_data t2 ON t1.cc_num = t2.cc_num AND t2.datetime >= t1.datetime - INTERVAL 24 HOURS AND t2.datetime < t1.datetime
    GROUP BY t1.transaction_id
""")
transactions_df = transactions_df.join(velocity_df, "transaction_id", "left").fillna({"txn_count_24h": 0})

# For score data, use combined data for velocity
combined_for_velocity = transactions_df.unionByName(score_df)
combined_for_velocity.createOrReplaceTempView("combined_txn")
score_velocity_df = spark.sql("""
    SELECT t1.transaction_id, COUNT(t2.transaction_id) - 1 as txn_count_24h
    FROM combined_txn t1
    LEFT JOIN combined_txn t2 ON t1.cc_num = t2.cc_num AND t2.datetime >= t1.datetime - INTERVAL 24 HOURS AND t2.datetime < t1.datetime
    GROUP BY t1.transaction_id
""")
score_df = score_df.join(score_velocity_df, "transaction_id", "left").fillna({"txn_count_24h": 0})

print("Feature engineering complete")

# Step 3: Create Feature Group
print("\\nCreating feature group...")
fs = FeatureStoreClient()

feature_df = transactions_df.select([
    "transaction_id", "cc_num", "datetime", "amount", "merchant", "category", 
    "lat", "long", "is_fraud", "hour_of_day", "day_of_week", "amount_log",
    "txn_count_24h", "card_avg_amount", "card_std_amount", "card_txn_count",
    "amount_dev_from_avg", "max_merchant_freq", "unique_merchants", 
    "max_category_freq", "geo_distance_from_avg", "card_avg_lat", "card_avg_long"
])

try:
    fs.create_table(name="{FG_FULL_NAME}", primary_keys=["transaction_id"], df=feature_df, description="Credit card transaction features")
    print(f"✓ Created feature group: {FG_FULL_NAME}")
except:
    fs.write_table(name="{FG_FULL_NAME}", df=feature_df, mode="overwrite")
    print(f"✓ Updated feature group: {FG_FULL_NAME}")

# Step 4: Create Training Dataset
print("\\nCreating training dataset...")
training_df = transactions_df.select([
    "transaction_id", "amount", "hour_of_day", "day_of_week", "amount_log",
    "txn_count_24h", "card_avg_amount", "card_std_amount", "card_txn_count",
    "amount_dev_from_avg", "max_merchant_freq", "unique_merchants", 
    "max_category_freq", "geo_distance_from_avg", "lat", "long", "is_fraud"
])

try:
    fs.create_table(name="{TD_FULL_NAME}", primary_keys=["transaction_id"], df=training_df, description="Training dataset")
    print(f"✓ Created training dataset: {TD_FULL_NAME}")
except:
    fs.write_table(name="{TD_FULL_NAME}", df=training_df, mode="overwrite")
    print(f"✓ Updated training dataset: {TD_FULL_NAME}")

# Step 5: Train Model
print("\\nTraining model...")

category_cols = ["merchant", "category"]
indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_idx", handleInvalid="keep").fit(training_df) for col in category_cols]
encoder = OneHotEncoder(inputCols=[f"{col}_idx" for col in category_cols], outputCols=[f"{col}_encoded" for col in category_cols])

numeric_cols = ["amount", "hour_of_day", "day_of_week", "amount_log", "txn_count_24h", 
               "card_avg_amount", "card_std_amount", "card_txn_count", "amount_dev_from_avg", 
               "max_merchant_freq", "unique_merchants", "max_category_freq", "geo_distance_from_avg", "lat", "long"]

assembler = VectorAssembler(inputCols=numeric_cols + [f"{col}_encoded" for col in category_cols], outputCol="features")

rf = RandomForestClassifier(labelCol="is_fraud", featuresCol="features", numTrees=200, maxDepth=6, seed=42)
pipeline = Pipeline(stages=[*indexers, encoder, assembler, rf])

train_data, test_data = training_df.randomSplit([0.8, 0.2], seed=42)
model = pipeline.fit(train_data)

predictions = model.transform(test_data)
evaluator = BinaryClassificationEvaluator(labelCol="is_fraud", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
auc = evaluator.evaluate(predictions)
print(f"✓ Test AUC: {auc}")

# Step 6: Register Model
print("\\nRegistering model...")
mlflow.set_experiment(f"/Users/{{current_user}}/{PREFIX}/fraud_detection")

with mlflow.start_run():
    mlflow.log_metric("auc", auc)
    model_path = mlflow.spark.log_model(model, "model", signature=infer_signature(train_data, model), registered_model_name="{MODEL_FULL_NAME}")
    result = mlflow.register_model(model_path, "{MODEL_FULL_NAME}")
    print(f"✓ Registered model: {MODEL_FULL_NAME}")

# Step 7: Score Transactions
print("\\nScoring transactions...")

score_feature_cols = ["amount", "hour_of_day", "day_of_week", "amount_log", "txn_count_24h", 
                     "card_avg_amount", "card_std_amount", "card_txn_count", "amount_dev_from_avg", 
                     "max_merchant_freq", "unique_merchants", "max_category_freq", "geo_distance_from_avg", 
                     "lat", "long", "merchant", "category", "transaction_id"]

score_data = score_df.select(score_feature_cols)
score_predictions = model.transform(score_data)

get_prob = udf(lambda v: float(v[1]) if v and len(v) > 1 else 0.0, FloatType())
result_df = score_predictions.select("transaction_id", get_prob("probability").alias("fraud_probability"))

result_df = result_df.withColumn("fraud_probability", 
                                F.when(F.col("fraud_probability") < 0, 0)
                                .when(F.col("fraud_probability") > 1, 1)
                                .otherwise(F.col("fraud_probability")))

print(f"✓ Scored {{result_df.count()}} transactions")

# Step 8: Create Predictions Feature Table
print("\\nCreating predictions table...")
try:
    fs.create_table(name="{PRED_FULL_NAME}", primary_keys=["transaction_id"], df=result_df, description="Fraud probability predictions")
    print(f"✓ Created predictions feature table: {PRED_FULL_NAME}")
except:
    fs.write_table(name="{PRED_FULL_NAME}", df=result_df, mode="overwrite")
    print(f"✓ Updated predictions feature table: {PRED_FULL_NAME}")

# Also create as regular table
result_df.write.saveAsTable("{PRED_FULL_NAME}", mode="overwrite")

# Create online table for low-latency lookup
try:
    online_table_name = "{PREFIX}_fraud_predictions_online"
    w.online_tables.create(name=online_table_name, source_table_name="{PRED_FULL_NAME}", comment="Online table for fraud probability lookup")
    print(f"✓ Created online table: {online_table_name}")
except Exception as e:
    print(f"✗ Error creating online table: {{e}}")

print(f"\\n=== PIPELINE COMPLETE ===")
print(f"Feature Group: {FG_FULL_NAME}")
print(f"Training Dataset: {TD_FULL_NAME}")
print(f"Model: {MODEL_FULL_NAME}")
print(f"Predictions: {PRED_FULL_NAME}")
print(f"Test AUC: {auc}")
"""

try:
    # Create a notebook with the pipeline script
    notebook_path = f"/Users/{current_user}/{PREFIX}/fraud_pipeline"
    w.workspace.upload(
        path=notebook_path,
        content=pipeline_script,
        language="PYTHON",
        overwrite=True
    )
    print(f"✓ Created notebook: {notebook_path}")
    
    # Execute the notebook on the cluster
    print("Executing notebook on cluster...")
    
    # Use the command execution API to run the notebook
    # First, let's try to run it as a job
    job_name = f"{PREFIX}_fraud_job"
    
    # Create the job
    job = w.jobs.create(
        name=job_name,
        tasks=[
            {
                "task_key": "fraud_pipeline",
                "notebook_task": {
                    "notebook_path": notebook_path
                },
                "existing_cluster_id": cluster_id
            }
        ]
    )
    print(f"✓ Created job: {job_name} (ID: {job.job_id})")
    
    # Run the job
    run = w.jobs.run_now(job_id=job.job_id)
    print(f"✓ Started job run: {run.run_id}")
    
    # Wait for completion
    print("Waiting for job to complete...")
    max_wait = 3600
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        time.sleep(30)
        run_info = w.jobs.get_run(run_id=run.run_id)
        state = run_info.state.life_cycle_state
        result = run_info.state.result_state if hasattr(run_info.state, 'result_state') else None
        
        print(f"  Job status: {state}, Result: {result}")
        
        if state == "TERMINATED":
            if result == "SUCCESS":
                print("✓ Job completed successfully!")
                break
            else:
                print(f"✗ Job failed with result: {result}")
                if hasattr(run_info.state, 'state_message'):
                    print(f"  Error: {run_info.state.state_message}")
                break
        elif state == "SKIPPED":
            print("✗ Job was skipped")
            break
    
    if time.time() - start_time >= max_wait:
        print("✗ Job timed out")
        
except Exception as e:
    print(f"✗ Error executing pipeline: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Pipeline execution complete ===")
