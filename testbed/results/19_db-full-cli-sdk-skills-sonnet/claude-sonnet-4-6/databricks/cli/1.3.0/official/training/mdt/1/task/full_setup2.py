# Databricks notebook source
# COMMAND ----------
import json
from pyspark.sql import functions as F

# Load raw data from volume
train_df = spark.read.csv(
    "/Volumes/workspace/mlpabc8d80a/data_upload/features_train.csv",
    header=True, inferSchema=True
)
serve_df = spark.read.csv(
    "/Volumes/workspace/mlpabc8d80a/data_upload/features_serve.csv",
    header=True, inferSchema=True
)

features = ["f1", "f2", "f3", "f4"]

# COMMAND ----------
# Compute population mean and std from training data only
stats = train_df.select([
    F.mean(c).alias(f"mean_{c}") for c in features
] + [
    F.stddev_pop(c).alias(f"std_{c}") for c in features
]).collect()[0]

# Standardize both splits using training stats
def standardize(df, features, stats):
    for c in features:
        mean_val = stats[f"mean_{c}"]
        std_val = stats[f"std_{c}"]
        df = df.withColumn(c, F.round((F.col(c) - mean_val) / std_val, 6))
    return df

train_scaled = standardize(train_df, features, stats).withColumn("split", F.lit("train"))
serve_scaled = standardize(serve_df, features, stats).withColumn("split", F.lit("serve"))

all_cols = ["row_id", "split", "f1", "f2", "f3", "f4"]
combined = train_scaled.select(all_cols).union(serve_scaled.select(all_cols))

# Register temp view
combined.createOrReplaceTempView("scaled_data_temp")

# COMMAND ----------
# Drop existing table and recreate
spark.sql("DROP TABLE IF EXISTS workspace.mlpabc8d80a.scaled7ecfaf")

spark.sql("""
    CREATE TABLE workspace.mlpabc8d80a.scaled7ecfaf (
        row_id STRING NOT NULL,
        split STRING,
        f1 DOUBLE,
        f2 DOUBLE,
        f3 DOUBLE,
        f4 DOUBLE,
        CONSTRAINT scaled7ecfaf_pk PRIMARY KEY (row_id)
    )
    TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
""")

# Insert using SQL from temp view
spark.sql("""
    INSERT INTO workspace.mlpabc8d80a.scaled7ecfaf
    SELECT row_id, split, f1, f2, f3, f4 FROM scaled_data_temp
""")

# COMMAND ----------
# Verify
count = spark.sql("SELECT COUNT(*) as cnt FROM workspace.mlpabc8d80a.scaled7ecfaf").collect()[0][0]
sample = spark.sql("SELECT * FROM workspace.mlpabc8d80a.scaled7ecfaf ORDER BY row_id LIMIT 3").collect()
sample_str = str([(r.row_id, r.split, round(r.f1, 4)) for r in sample])

dbutils.notebook.exit(json.dumps({
    "count_in_table": count,
    "sample": sample_str
}))
