# Databricks notebook source
# COMMAND ----------
# Load training and serving data
train_df = spark.read.csv(
    "/Volumes/workspace/mlpabc8d80a/data_upload/features_train.csv",
    header=True, inferSchema=True
)
serve_df = spark.read.csv(
    "/Volumes/workspace/mlpabc8d80a/data_upload/features_serve.csv",
    header=True, inferSchema=True
)

# COMMAND ----------
# Compute mean and population std from training data only
from pyspark.sql import functions as F

features = ["f1", "f2", "f3", "f4"]

# Get stats from training data
stats = train_df.select([
    F.mean(c).alias(f"mean_{c}") for c in features
] + [
    F.stddev_pop(c).alias(f"std_{c}") for c in features
]).collect()[0]

print("Training stats:")
for c in features:
    print(f"  {c}: mean={stats[f'mean_{c}']}, std={stats[f'std_{c}']}")

# COMMAND ----------
# Standardize both splits using training stats
def standardize(df, features, stats):
    for c in features:
        mean_val = stats[f"mean_{c}"]
        std_val = stats[f"std_{c}"]
        df = df.withColumn(c, F.round((F.col(c) - mean_val) / std_val, 6))
    return df

train_scaled = standardize(train_df, features, stats).withColumn("split", F.lit("train"))
serve_scaled = standardize(serve_df, features, stats).withColumn("split", F.lit("serve"))

# COMMAND ----------
# Combine both splits and select columns in order
combined = train_scaled.union(serve_scaled.select(train_scaled.columns))
combined = combined.select("row_id", "split", "f1", "f2", "f3", "f4")

print(f"Total rows: {combined.count()}")
combined.show(5)

# COMMAND ----------
# Write as Delta table (feature table)
table_name = "workspace.mlpabc8d80a.scaled7ecfaf"

combined.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(table_name)

print(f"Table {table_name} created successfully")

# COMMAND ----------
# Set table properties for feature store compatibility
spark.sql(f"""
ALTER TABLE {table_name}
SET TBLPROPERTIES (
    'delta.enableChangeDataFeed' = 'true'
)
""")

print("Change data feed enabled")

# COMMAND ----------
# Verify table
result = spark.sql(f"SELECT * FROM {table_name} LIMIT 5")
result.show()
print(f"Total rows in table: {spark.sql(f'SELECT COUNT(*) FROM {table_name}').collect()[0][0]}")
