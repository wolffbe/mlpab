# Databricks notebook source
# MAGIC %md
# MAGIC Load data, compute standardization, and register feature table.

# COMMAND ----------

# Read data from volume
train_df = spark.read.csv("/Volumes/workspace/mlpab8271b0/uploads/train.csv", header=True, inferSchema=True)
serve_df = spark.read.csv("/Volumes/workspace/mlpab8271b0/uploads/serve.csv", header=True, inferSchema=True)

# Compute means and stds from training data only
from pyspark.sql.functions import avg, stddev_pop
stats = train_df.select(
    avg("f1").alias("mean_f1"),
    avg("f2").alias("mean_f2"),
    avg("f3").alias("mean_f3"),
    avg("f4").alias("mean_f4"),
    stddev_pop("f1").alias("std_f1"),
    stddev_pop("f2").alias("std_f2"),
    stddev_pop("f3").alias("std_f3"),
    stddev_pop("f4").alias("std_f4")
).collect()[0]

# Standardize both splits
from pyspark.sql.functions import col, lit, round

def standardize(df, split_name):
    return df.select(
        "row_id",
        lit(split_name).alias("split"),
        round((col("f1") - stats["mean_f1"]) / stats["std_f1"], 6).alias("f1"),
        round((col("f2") - stats["mean_f2"]) / stats["std_f2"], 6).alias("f2"),
        round((col("f3") - stats["mean_f3"]) / stats["std_f3"], 6).alias("f3"),
        round((col("f4") - stats["mean_f4"]) / stats["std_f4"], 6).alias("f4")
    )

standardized_train = standardize(train_df, "train")
standardized_serve = standardize(serve_df, "serve")

# Combine and save
from pyspark.sql import DataFrame
combined = standardized_train.union(standardized_serve)
combined.write.saveAsTable("workspace.mlpab8271b0.scaled205aa4")

# Register feature table
spark.sql("""
CREATE FEATURE TABLE IF NOT EXISTS workspace.mlpab8271b0.scaled205aa4_v1 (
  row_id STRING,
  split STRING,
  f1 DOUBLE,
  f2 DOUBLE,
  f3 DOUBLE,
  f4 DOUBLE
) TBLPROPERTIES (
  'updates.mode' = 'APPEND_ONLY'
)
""")

# Enable online access
spark.sql("CREATE ONLINE TABLE IF NOT EXISTS workspace.mlpab8271b0.scaled205aa4_online_v1 AS SELECT * FROM workspace.mlpab8271b0.scaled205aa4")