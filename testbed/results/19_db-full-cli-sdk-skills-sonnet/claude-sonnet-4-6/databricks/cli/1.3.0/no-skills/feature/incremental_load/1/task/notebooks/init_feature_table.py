# Databricks notebook source
# MAGIC %md
# MAGIC # Initialize Feature Table: incremental3526e9

# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering -q

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

catalog = "workspace"
schema = "mlpab312fe6"
table_name = "incremental3526e9"
full_table_name = f"{catalog}.{schema}.{table_name}"
volume_path = f"/Volumes/{catalog}/{schema}/mlpab312fe6_data"

print(f"Full table name: {full_table_name}")
print(f"Volume path: {volume_path}")

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, LongType, DoubleType

schema_struct = StructType([
    StructField("row_id", StringType(), True),
    StructField("account_id", StringType(), True),
    StructField("event_time", LongType(), True),
    StructField("amount", DoubleType(), True),
    StructField("category", StringType(), True),
])

df = spark.read \
    .option("header", "true") \
    .schema(schema_struct) \
    .csv(f"{volume_path}/increment_*.csv")

print(f"Total rows loaded: {df.count()}")
df.show(5)

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

# Record key: row_id; event_time (bigint, epoch ms) is a feature column
# timeseries_column requires TIMESTAMP type which conflicts with epoch ms bigint
fe.create_table(
    name=full_table_name,
    primary_keys=["row_id"],
    df=df,
    description="Events feature table. Record key: row_id, event-time column: event_time (bigint, epoch milliseconds)."
)

print(f"Feature table {full_table_name} created and initial data loaded successfully")

# COMMAND ----------

row_count = spark.table(full_table_name).count()
print(f"Feature table {full_table_name} is ready with {row_count} rows")
