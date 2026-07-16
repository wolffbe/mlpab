#!/usr/bin/env python3
"""
Databricks job script to create the training dataset.
This script will be uploaded and executed as a job.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, row_number
from pyspark.sql.window import Window
import os
import mlflow

# Environment variables
SCHEMA_NAME = os.environ["MLPAB_DATABRICKS_SCHEMA"].replace(".", "_")
DATA_DIR = "/dbfs{}".format(os.path.abspath("./data"))

# Initialize Spark
spark = SparkSession.builder.getOrCreate()

# Create schema
spark.sql(f"CREATE SCHEMA IF NOT EXISTS workspace.{SCHEMA_NAME}")

# Read CSV files
transactions_df = spark.read.csv(f"{DATA_DIR}/transactions.csv", header=True, inferSchema=True)
transactions_late_df = spark.read.csv(f"{DATA_DIR}/transactions_late.csv", header=True, inferSchema=True)
profiles_df = spark.read.csv(f"{DATA_DIR}/profiles.csv", header=True, inferSchema=True)
activity_df = spark.read.csv(f"{DATA_DIR}/activity.csv", header=True, inferSchema=True)
account_health_df = spark.read.csv(f"{DATA_DIR}/account_health.csv", header=True, inferSchema=True)
labels_df = spark.read.csv(f"{DATA_DIR}/labels.csv", header=True, inferSchema=True)

# Register DataFrames as temporary views
transactions_df.createOrReplaceTempView("transactions")
transactions_late_df.createOrReplaceTempView("transactions_late")
profiles_df.createOrReplaceTempView("profiles")
activity_df.createOrReplaceTempView("activity")
account_health_df.createOrReplaceTempView("account_health")
labels_df.createOrReplaceTempView("labels")

# Create the training dataset
query = f"""
CREATE OR REPLACE TABLE workspace.{SCHEMA_NAME}.churntrainingaf8b21_v1 AS
WITH latest_transactions AS (
    SELECT 
        t.account_id,
        t.amount,
        t.balance,
        l.label_time
    FROM labels l
    LEFT JOIN (
        SELECT 
            account_id,
            amount,
            balance,
            event_time
        FROM (
            SELECT 
                account_id,
                amount,
                balance,
                event_time,
                ROW_NUMBER() OVER (PARTITION BY account_id ORDER BY event_time DESC) as rn
            FROM (
                SELECT * FROM transactions
                UNION ALL
                SELECT * FROM transactions_late
            )
        )
        WHERE rn = 1
    ) t
    ON l.account_id = t.account_id AND t.event_time <= l.label_time
),
latest_profiles AS (
    SELECT 
        p.account_id,
        p.credit_score,
        p.tier,
        l.label_time
    FROM labels l
    LEFT JOIN (
        SELECT 
            account_id,
            credit_score,
            tier,
            event_time
        FROM (
            SELECT 
                account_id,
                credit_score,
                tier,
                event_time,
                ROW_NUMBER() OVER (PARTITION BY account_id ORDER BY event_time DESC) as rn
            FROM profiles
        )
        WHERE rn = 1
    ) p
    ON l.account_id = p.account_id AND p.event_time <= l.label_time
),
latest_activity AS (
    SELECT 
        a.account_id,
        a.sessions_7d,
        l.label_time
    FROM labels l
    LEFT JOIN (
        SELECT 
            account_id,
            sessions_7d,
            event_time
        FROM (
            SELECT 
                account_id,
                sessions_7d,
                event_time,
                ROW_NUMBER() OVER (PARTITION BY account_id ORDER BY event_time DESC) as rn
            FROM activity
        )
        WHERE rn = 1
    ) a
    ON l.account_id = a.account_id AND a.event_time <= l.label_time
),
latest_health AS (
    SELECT 
        h.account_id,
        h.health_score,
        l.label_time
    FROM labels l
    LEFT JOIN (
        SELECT 
            account_id,
            health_score,
            event_time
        FROM (
            SELECT 
                account_id,
                health_score,
                event_time,
                ROW_NUMBER() OVER (PARTITION BY account_id ORDER BY event_time DESC) as rn
            FROM account_health
        )
        WHERE rn = 1
    ) h
    ON l.account_id = h.account_id AND h.event_time <= l.label_time
)
SELECT 
    l.account_id,
    l.label_time,
    t.amount,
    t.balance,
    p.credit_score,
    p.tier,
    a.sessions_7d,
    h.health_score,
    l.churned
FROM labels l
LEFT JOIN latest_transactions t ON l.account_id = t.account_id AND l.label_time = t.label_time
LEFT JOIN latest_profiles p ON l.account_id = p.account_id AND l.label_time = p.label_time
LEFT JOIN latest_activity a ON l.account_id = a.account_id AND l.label_time = a.label_time
LEFT JOIN latest_health h ON l.account_id = h.account_id AND l.label_time = h.label_time
"""

spark.sql(query)
print("Training dataset created: churntrainingaf8b21_v1")

# Register the dataset as a versioned model
mlflow.set_registry_uri("databricks")
model_uri = f"runs:/dummy_run_id/workspace.{SCHEMA_NAME}.churntrainingaf8b21_v1"
mlflow.register_model(model_uri, f"workspace.{SCHEMA_NAME}.churntrainingaf8b21")
print("Registered model version: churntrainingaf8b21, version 1")