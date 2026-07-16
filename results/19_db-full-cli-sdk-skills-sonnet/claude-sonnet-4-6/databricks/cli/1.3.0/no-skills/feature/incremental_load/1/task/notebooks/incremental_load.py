# Databricks notebook source
# MAGIC %md
# MAGIC # Incremental Load Job: incrementaljob3526e9
# MAGIC Runs daily to ingest new increment files into the feature table.
# MAGIC Uses COPY INTO which tracks processed files and only loads new ones.

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

print(f"Running incremental load for: {full_table_name}")
print(f"Source volume: {volume_path}")

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, LongType, DoubleType

schema_struct = StructType([
    StructField("row_id", StringType(), True),
    StructField("account_id", StringType(), True),
    StructField("event_time", LongType(), True),
    StructField("amount", DoubleType(), True),
    StructField("category", StringType(), True),
])

# Read all CSV files from volume (new ones will be picked up automatically)
df_new = spark.read \
    .option("header", "true") \
    .schema(schema_struct) \
    .csv(f"{volume_path}/increment_*.csv")

print(f"Files found: {df_new.count()} rows")

# COMMAND ----------

# Use Feature Engineering write_table with merge mode
# MERGE on row_id (primary key): no-ops for existing rows, inserts new rows
# This is idempotent - safe to run multiple times
from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

fe.write_table(
    name=full_table_name,
    df=df_new,
    mode="merge"
)

print(f"Incremental load completed. Current row count: {spark.table(full_table_name).count()}")
