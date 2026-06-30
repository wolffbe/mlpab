# Databricks notebook source
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, DoubleType
spark = SparkSession.builder.getOrCreate()

vol_path = "/Volumes/workspace/mlpab6c8eeb/mlpab6c8eeb_data"
log = {}

# Drop existing table
spark.sql("DROP TABLE IF EXISTS workspace.mlpab6c8eeb.predictions7b586d")

# Read predictions and create feature table with primary key
df = spark.read.csv(f"{vol_path}/predictions.csv", header=True, inferSchema=True)
df.createOrReplaceTempView("predictions_tmp")

spark.sql("""
    CREATE TABLE workspace.mlpab6c8eeb.predictions7b586d
    USING DELTA
    AS SELECT CAST(row_id AS STRING) AS row_id, CAST(score AS DOUBLE) AS score
    FROM predictions_tmp
""")
spark.sql("ALTER TABLE workspace.mlpab6c8eeb.predictions7b586d ALTER COLUMN row_id SET NOT NULL")
spark.sql("ALTER TABLE workspace.mlpab6c8eeb.predictions7b586d ADD CONSTRAINT predictions7b586d_pk PRIMARY KEY (row_id)")
log["feature_table"] = "created with primary key"

# Check available feature engineering packages
for pkg in [
    "databricks.feature_engineering",
    "databricks.feature_store",
    "mlflow.feature_store",
]:
    try:
        mod = __import__(pkg, fromlist=[""])
        log[pkg] = str(dir(mod))[:200]
    except Exception as e:
        log[pkg] = f"NOT available: {e}"

# Write log to volume
with open(f"{vol_path}/register_log.json", "w") as f:
    json.dump(log, f, indent=2)

dbutils.notebook.exit(json.dumps(log))
