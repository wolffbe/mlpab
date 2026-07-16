#!/usr/bin/env python3
"""
Full FTI pipeline for credit card fraud detection.
Uses Databricks SDK to create and execute everything on the platform.
"""
import os
import sys

# Set up environment
os.environ['DATABRICKS_HOST'] = os.getenv('DATABRICKS_HOST', '***REDACTED***')
os.environ['DATABRICKS_TOKEN'] = os.getenv('DATABRICKS_TOKEN')

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog, compute, sql, jobs, ml

def main():
    w = WorkspaceClient()
    
    # Configuration
    SCHEMA = os.getenv('MLPAB_DATABRICKS_SCHEMA', 'workspace.mlpab12289f')
    PREFIX = os.getenv('MLPAB_DATABRICKS_PREFIX', 'mlpab12289f')
    
    # Split schema into catalog and schema name
    catalog_name, schema_name = SCHEMA.split('.')
    
    print(f"Using catalog: {catalog_name}, schema: {schema_name}")
    
    # Step 1: Upload CSV files to DBFS
    print("\n=== Step 1: Uploading CSV files to DBFS ===")
    dbfs_path = f"dbfs:/FileStore/{PREFIX}"
    
    # Upload transactions.csv
    with open('data/transactions.csv', 'rb') as f:
        w.dbfs.upload(f'{dbfs_path}/transactions.csv', f.read(), overwrite=True)
    print("Uploaded transactions.csv")
    
    # Upload score_transactions.csv
    with open('data/score_transactions.csv', 'rb') as f:
        w.dbfs.upload(f'{dbfs_path}/score_transactions.csv', f.read(), overwrite=True)
    print("Uploaded score_transactions.csv")
    
    # Step 2: Create raw tables from CSV files
    print("\n=== Step 2: Creating raw tables ===")
    
    # Create transactions table
    w.statement_execution.execute_statement(
        warehouse_id='a832b544eb7dc3fe',  # Serverless Starter Warehouse
        catalog=catalog_name,
        schema=schema_name,
        statement=f"""
        CREATE TABLE IF NOT EXISTS {catalog_name}.{schema_name}.raw_transactions (
            transaction_id STRING,
            cc_num STRING,
            datetime TIMESTAMP,
            amount DOUBLE,
            merchant STRING,
            category STRING,
            lat DOUBLE,
            long DOUBLE,
            is_fraud INT
        ) USING CSV
        OPTIONS (
            path '{dbfs_path}/transactions.csv',
            header 'true',
            inferSchema 'true'
        )
        """
    )
    print("Created raw_transactions table")
    
    # Create score_transactions table
    w.statement_execution.execute_statement(
        warehouse_id='a832b544eb7dc3fe',
        catalog=catalog_name,
        schema=schema_name,
        statement=f"""
        CREATE TABLE IF NOT EXISTS {catalog_name}.{schema_name}.raw_score_transactions (
            transaction_id STRING,
            cc_num STRING,
            datetime TIMESTAMP,
            amount DOUBLE,
            merchant STRING,
            category STRING,
            lat DOUBLE,
            long DOUBLE
        ) USING CSV
        OPTIONS (
            path '{dbfs_path}/score_transactions.csv',
            header 'true',
            inferSchema 'true'
        )
        """
    )
    print("Created raw_score_transactions table")
    
    # Step 3: Create feature engineering SQL
    print("\n=== Step 3: Feature Engineering ===")
    
    # First, create a table with card profiles (average location per card)
    w.statement_execution.execute_statement(
        warehouse_id='a832b544eb7dc3fe',
        catalog=catalog_name,
        schema=schema_name,
        statement=f"""
        CREATE OR REPLACE TABLE {catalog_name}.{schema_name}.card_profiles AS
        SELECT 
            cc_num,
            AVG(lat) as avg_lat,
            AVG(long) as avg_long,
            STDDEV(lat) as lat_std,
            STDDEV(long) as long_std,
            COUNT(*) as txn_count,
            AVG(amount) as avg_amount,
            STDDEV(amount) as amount_std,
            MIN(datetime) as first_txn,
            MAX(datetime) as last_txn
        FROM {catalog_name}.{schema_name}.raw_transactions
        GROUP BY cc_num
        """
    )
    print("Created card_profiles table")
    
    # Create feature table with engineered features
    w.statement_execution.execute_statement(
        warehouse_id='a832b544eb7dc3fe',
        catalog=catalog_name,
        schema=schema_name,
        statement=f"""
        CREATE OR REPLACE TABLE {catalog_name}.{schema_name}.cctxn015310 AS
        SELECT 
            t.transaction_id,
            t.cc_num,
            t.datetime,
            t.amount,
            t.merchant,
            t.category,
            t.lat,
            t.long,
            t.is_fraud,
            -- Time-based features
            EXTRACT(HOUR FROM t.datetime) as hour_of_day,
            EXTRACT(DAYOFWEEK FROM t.datetime) as day_of_week,
            -- Card velocity features (from window functions)
            COUNT(*) OVER (PARTITION BY t.cc_num ORDER BY t.datetime 
                           RANGE BETWEEN INTERVAL 1 HOUR PRECEDING AND CURRENT ROW) as txn_last_hour,
            COUNT(*) OVER (PARTITION BY t.cc_num ORDER BY t.datetime 
                           RANGE BETWEEN INTERVAL 24 HOURS PRECEDING AND CURRENT ROW) as txn_last_24h,
            SUM(t.amount) OVER (PARTITION BY t.cc_num ORDER BY t.datetime 
                               RANGE BETWEEN INTERVAL 1 HOUR PRECEDING AND CURRENT ROW) as amount_last_hour,
            SUM(t.amount) OVER (PARTITION BY t.cc_num ORDER BY t.datetime 
                               RANGE BETWEEN INTERVAL 24 HOURS PRECEDING AND CURRENT ROW) as amount_last_24h,
            -- Geo distance from card's average location
            CASE 
                WHEN p.avg_lat IS NOT NULL AND p.avg_long IS NOT NULL 
                THEN 6371 * 2 * ASIN(SQRT(
                    POWER(SIN((RADIANS(t.lat) - RADIANS(p.avg_lat)) / 2), 2) +
                    COS(RADIANS(t.lat)) * COS(RADIANS(p.avg_lat)) *
                    POWER(SIN((RADIANS(t.long) - RADIANS(p.avg_long)) / 2), 2)
                ))
                ELSE NULL
            END as geo_distance_km,
            -- Amount deviation from card's average
            CASE 
                WHEN p.avg_amount > 0 THEN (t.amount - p.avg_amount) / NULLIF(p.amount_std, 0)
                ELSE NULL
            END as amount_z_score,
            -- Category encoding
            CASE t.category
                WHEN 'fuel' THEN 1
                WHEN 'travel' THEN 2
                WHEN 'online' THEN 3
                WHEN 'restaurant' THEN 4
                WHEN 'health' THEN 5
                WHEN 'cash_advance' THEN 6
                WHEN 'electronics' THEN 7
                WHEN 'entertainment' THEN 8
                WHEN 'grocery' THEN 9
                WHEN 'clothing' THEN 10
                ELSE 0
            END as category_encoded,
            -- Time since last transaction for this card
            t.datetime - LAG(t.datetime) OVER (PARTITION BY t.cc_num ORDER BY t.datetime) as time_since_last_txn
        FROM {catalog_name}.{schema_name}.raw_transactions t
        LEFT JOIN {catalog_name}.{schema_name}.card_profiles p ON t.cc_num = p.cc_num
        """
    )
    print("Created feature table cctxn015310")
    
    # Step 4: Create training dataset
    print("\n=== Step 4: Creating training dataset ===")
    
    w.statement_execution.execute_statement(
        warehouse_id='a832b544eb7dc3fe',
        catalog=catalog_name,
        schema=schema_name,
        statement=f"""
        CREATE OR REPLACE TABLE {catalog_name}.{schema_name}.cctd015310 AS
        SELECT 
            transaction_id,
            cc_num,
            datetime,
            amount,
            merchant,
            category,
            lat,
            long,
            is_fraud,
            hour_of_day,
            day_of_week,
            COALESCE(txn_last_hour, 0) as txn_last_hour,
            COALESCE(txn_last_24h, 0) as txn_last_24h,
            COALESCE(amount_last_hour, 0) as amount_last_hour,
            COALESCE(amount_last_24h, 0) as amount_last_24h,
            COALESCE(geo_distance_km, 0) as geo_distance_km,
            COALESCE(amount_z_score, 0) as amount_z_score,
            COALESCE(category_encoded, 0) as category_encoded,
            COALESCE(UNIX_TIMESTAMP(time_since_last_txn), 0) as time_since_last_txn_sec
        FROM {catalog_name}.{schema_name}.cctxn015310
        WHERE is_fraud IS NOT NULL
        """
    )
    print("Created training dataset cctd015310")
    
    # Step 5: Train model using MLlib (via SQL)
    # We'll use Logistic Regression from MLlib
    print("\n=== Step 5: Training model ===")
    
    # First, prepare the data for MLlib
    w.statement_execution.execute_statement(
        warehouse_id='a832b544eb7dc3fe',
        catalog=catalog_name,
        schema=schema_name,
        statement=f"""
        CREATE OR REPLACE TABLE {catalog_name}.{schema_name}.train_data AS
        SELECT 
            is_fraud as label,
            array(
                amount,
                hour_of_day,
                day_of_week,
                txn_last_hour,
                txn_last_24h,
                amount_last_hour,
                amount_last_24h,
                geo_distance_km,
                amount_z_score,
                category_encoded,
                time_since_last_txn_sec
            ) as features
        FROM {catalog_name}.{schema_name}.cctd015310
        WHERE is_fraud IS NOT NULL
        """
    )
    print("Created train_data table")
    
    # Train logistic regression model
    # Note: This might need to be done via a notebook or job
    # Let's try using the ML API
    print("\n=== Attempting to train via ML API ===")
    
    # Create a notebook with the training code
    notebook_path = f"/Users/{w.current_user.me().emails[0]}/{PREFIX}/train_model"
    notebook_content = """# Databricks notebook source
# MAGIC %md ## Credit Card Fraud Detection Model Training

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import col

# COMMAND ----------

catalog_name = "workspace"
schema_name = "mlpab12289f"
full_table = f"{catalog_name}.{schema_name}.train_data"

# COMMAND ----------

df = spark.table(full_table)
print(f"Training data count: {df.count()}")
print(f"Fraud ratio: {df.filter(col('label') == 1).count() / df.count()}")

# COMMAND ----------

# Split data
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# COMMAND ----------

# Create pipeline
assembler = VectorAssembler(
    inputCols=["features"],
    outputCol="raw_features"
)

scaler = StandardScaler(
    inputCol="raw_features",
    outputCol="scaled_features",
    withStd=True,
    withMean=True
)

lr = LogisticRegression(
    featuresCol="scaled_features",
    labelCol="label",
    predictionCol="prediction",
    probabilityCol="probability",
    maxIter=100,
    regParam=0.01,
    elasticNetParam=0.8
)

pipeline = Pipeline(stages=[assembler, scaler, lr])

# COMMAND ----------

# Train model
model = pipeline.fit(train_df)

# COMMAND ----------

# Evaluate
predictions = model.transform(test_df)
evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)
roc_auc = evaluator.evaluate(predictions)
print(f"Test ROC AUC: {roc_auc}")

# COMMAND ----------

# Save model
model_path = f"dbfs:/FileStore/{os.getenv('MLPAB_DATABRICKS_PREFIX', 'mlpab12289f')}/ccmodel015310"
model.write().overwrite().save(model_path)
print(f"Model saved to {model_path}")

# COMMAND ----------

# Register model in MLflow
import mlflow
import os

model_name = "ccmodel015310"
model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"

# Log metrics
mlflow.log_metric("roc_auc", roc_auc)
mlflow.log_param("model_type", "LogisticRegression")

# Register model
result = mlflow.register_model(model_uri, model_name)
print(f"Model registered: {result}")

# COMMAND ----------

# Also register via Databricks SDK
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

# Create registered model
try:
    w.registered_models.create(
        name=model_name,
        description="Credit Card Fraud Detection Model"
    )
except:
    pass  # Already exists

# Create model version
w.model_versions.create(
    name=model_name,
    source=f"dbfs:/FileStore/{os.getenv('MLPAB_DATABRICKS_PREFIX', 'mlpab12289f')}/ccmodel015310",
    run_id=mlflow.active_run().info.run_id,
    comment=f"ROC AUC: {roc_auc}"
)
"""
    
    # Write notebook
    w.workspace.upload(notebook_path + ".py", notebook_content.encode(), overwrite=True)
    print(f"Created notebook at {notebook_path}")
    
    # Create a cluster for training
    print("\n=== Creating cluster for training ===")
    cluster_name = f"{PREFIX}_train_cluster"
    
    try:
        cluster = w.clusters.get(cluster_name)
        print(f"Cluster already exists: {cluster.state}")
    except:
        # Create new cluster
        cluster = w.clusters.create(
            cluster_name=cluster_name,
            node_type_id="Standard_DS3_v2",
            spark_version="14.3.x-scala2.12",
            num_workers=2,
            autoscale=compute.ClusterAutoscale(
                min_workers=2,
                max_workers=4
            ),
            spark_conf={
                "spark.databricks.delta.preview.enabled": "true"
            }
        )
        print(f"Created cluster: {cluster.cluster_id}")
        
        # Wait for cluster to be ready
        print("Waiting for cluster to start...")
        w.clusters.wait_get_cluster_running(cluster.cluster_id, timeout=300)
        print("Cluster is running!")
    
    # Submit job to run the notebook
    print("\n=== Submitting training job ===")
    job_name = f"{PREFIX}_train_fraud_model"
    
    job = w.jobs.create(
        name=job_name,
        tasks=[
            jobs.Task(
                task_key="train",
                notebook_task=jobs.NotebookTask(
                    notebook_path=notebook_path + ".py"
                ),
                existing_cluster_id=cluster.cluster_id
            )
        ]
    )
    print(f"Created job: {job.job_id}")
    
    # Run and wait
    run = w.jobs.run_now_and_wait(job.job_id, timeout=600)
    print(f"Job run completed: {run.state}")
    
    if run.state != jobs.RunState.LIFE_CYCLE_STATE_TERMINATED:
        print(f"Job failed with result state: {run.result_state}")
        return 1
    
    print("Training job completed successfully!")
    
    # Step 6: Score the transactions
    print("\n=== Step 6: Scoring transactions ===")
    
    # Create scoring notebook
    score_notebook_path = f"/Users/{w.current_user.me().emails[0]}/{PREFIX}/score_model"
    score_notebook_content = """# Databricks notebook source
# MAGIC %md ## Score Transactions

# COMMAND ----------

from pyspark.ml import PipelineModel
from pyspark.sql.functions import col, when, lit
import os

# COMMAND ----------

catalog_name = "workspace"
schema_name = "mlpab12289f"
PREFIX = os.getenv('MLPAB_DATABRICKS_PREFIX', 'mlpab12289f')

# COMMAND ----------

# Load model
model_path = f"dbfs:/FileStore/{PREFIX}/ccmodel015310"
model = PipelineModel.load(model_path)

# COMMAND ----------

# Prepare score data
# First, create features for score transactions
score_table = f"{catalog_name}.{schema_name}.raw_score_transactions"

# Join with card profiles
card_profiles_table = f"{catalog_name}.{schema_name}.card_profiles"

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.window import Window

# Load score data
df = spark.table(score_table)

# Join with card profiles
profiles = spark.table(card_profiles_table)
df = df.join(profiles, on="cc_num", how="left")

# Add features
df = df.withColumn("hour_of_day", hour(col("datetime")))
df = df.withColumn("day_of_week", dayofweek(col("datetime")))

# Window functions for velocity
window_spec_hour = Window.partitionBy("cc_num").orderBy("datetime").rangeBetween(-3600, 0)
window_spec_day = Window.partitionBy("cc_num").orderBy("datetime").rangeBetween(-86400, 0)

df = df.withColumn("txn_last_hour", count("*").over(window_spec_hour))
df = df.withColumn("txn_last_24h", count("*").over(window_spec_day))
df = df.withColumn("amount_last_hour", sum("amount").over(window_spec_hour))
df = df.withColumn("amount_last_24h", sum("amount").over(window_spec_day))

# Geo distance
def haversine(lat1, lon1, lat2, lon2):
    from math import radians, sin, cos, sqrt, asin
    R = 6371  # Earth radius in km
    dLat = radians(lat2 - lat1)
    dLon = radians(lon2 - lon1)
    a = sin(dLat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dLon/2)**2
    return 2 * R * asin(sqrt(a))

from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf

haversine_udf = udf(haversine, DoubleType())

df = df.withColumn(
    "geo_distance_km",
    when(
        (col("avg_lat").isNotNull()) & (col("avg_long").isNotNull()),
        haversine_udf(col("lat"), col("long"), col("avg_lat"), col("avg_long"))
    ).otherwise(lit(0))
)

df = df.withColumn(
    "amount_z_score",
    when(
        (col("avg_amount").isNotNull()) & (col("amount_std").isNotNull()) & (col("amount_std") > 0),
        (col("amount") - col("avg_amount")) / col("amount_std")
    ).otherwise(lit(0))
)

# Category encoding
category_mapping = {
    'fuel': 1, 'travel': 2, 'online': 3, 'restaurant': 4, 'health': 5,
    'cash_advance': 6, 'electronics': 7, 'entertainment': 8, 'grocery': 9, 'clothing': 10
}

def encode_category(cat):
    return category_mapping.get(cat, 0)

from pyspark.sql.types import IntegerType
encode_category_udf = udf(encode_category, IntegerType())

df = df.withColumn("category_encoded", encode_category_udf(col("category")))

# Time since last transaction
df = df.withColumn(
    "time_since_last_txn_sec",
    unix_timestamp(col("datetime")) - lag(unix_timestamp(col("datetime"))).over(Window.partitionBy("cc_num").orderBy("datetime"))
)
df = df.fillna(0, subset=["time_since_last_txn_sec"])

# COMMAND ----------

# Select features in same order as training
feature_cols = [
    "amount", "hour_of_day", "day_of_week", "txn_last_hour", "txn_last_24h",
    "amount_last_hour", "amount_last_24h", "geo_distance_km", 
    "amount_z_score", "category_encoded", "time_since_last_txn_sec"
]

# Create features array
df = df.withColumn("features", array(*[col(c) for c in feature_cols]))

# COMMAND ----------

# Score
predictions = model.transform(df)

# COMMAND ----------

# Save predictions
output_table = f"{catalog_name}.{schema_name}.ccpred015310"
predictions.select(
    col("transaction_id"),
    col("probability")[1].alias("fraud_probability")
).write.mode("overwrite").saveAsTable(output_table)

print(f"Predictions saved to {output_table}")

# COMMAND ----------

# Also create as online table for low-latency lookup
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

w.feature_store.publish_table(
    name=output_table,
    catalog_name=catalog_name,
    schema_name=schema_name
)

print("Published predictions table for online lookup")
"""
    
    w.workspace.upload(score_notebook_path + ".py", score_notebook_content.encode(), overwrite=True)
    print(f"Created scoring notebook at {score_notebook_path}")
    
    # Submit scoring job
    score_job_name = f"{PREFIX}_score_fraud_model"
    
    try:
        score_job = w.jobs.get(score_job_name)
        print(f"Score job already exists: {score_job.job_id}")
    except:
        score_job = w.jobs.create(
            name=score_job_name,
            tasks=[
                jobs.Task(
                    task_key="score",
                    notebook_task=jobs.NotebookTask(
                        notebook_path=score_notebook_path + ".py"
                    ),
                    existing_cluster_id=cluster.cluster_id
                )
            ]
        )
        print(f"Created score job: {score_job.job_id}")
    
    # Run and wait
    score_run = w.jobs.run_now_and_wait(score_job.job_id, timeout=600)
    print(f"Score job run completed: {score_run.state}")
    
    if score_run.state != jobs.RunState.LIFE_CYCLE_STATE_TERMINATED:
        print(f"Score job failed with result state: {score_run.result_state}")
        return 1
    
    print("Scoring job completed successfully!")
    
    # Step 7: Make predictions table available for online lookup
    print("\n=== Step 7: Publishing for online lookup ===")
    
    # The table should already be published via the notebook
    # Let's verify it exists
    try:
        table = w.tables.get(catalog_name, schema_name, "ccpred015310")
        print(f"Predictions table exists: {table.name}")
    except Exception as e:
        print(f"Error checking table: {e}")
    
    print("\n=== Pipeline Complete ===")
    print(f"Feature group: {catalog_name}.{schema_name}.cctxn015310")
    print(f"Training dataset: {catalog_name}.{schema_name}.cctd015310")
    print(f"Model: ccmodel015310")
    print(f"Predictions: {catalog_name}.{schema_name}.ccpred015310")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
