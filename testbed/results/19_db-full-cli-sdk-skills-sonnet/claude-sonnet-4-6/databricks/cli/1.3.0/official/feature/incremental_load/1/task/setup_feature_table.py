# Databricks notebook source
# MAGIC %pip install databricks-feature-engineering --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

catalog = "workspace"
schema = "mlpabf12520"
table_name = "incremental3526e9"
full_table_name = f"{catalog}.{schema}.{table_name}"
volume_path = f"/Volumes/{catalog}/{schema}/incremental_data"

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, LongType, DoubleType
from pyspark.sql.functions import col, from_unixtime, to_timestamp

schema_def = StructType([
    StructField("row_id", StringType(), False),
    StructField("account_id", StringType(), True),
    StructField("event_time", LongType(), True),
    StructField("amount", DoubleType(), True),
    StructField("category", StringType(), True),
])

# COMMAND ----------

dfs = []
for i in range(1, 7):
    path = f"{volume_path}/increment_{i:02d}.csv"
    df = spark.read.csv(path, header=True, schema=schema_def)
    dfs.append(df)

from functools import reduce
all_df = reduce(lambda a, b: a.union(b), dfs)

# Convert epoch milliseconds to TIMESTAMP (Feature Engineering requires TIMESTAMP type)
all_df = all_df.withColumn(
    "event_time",
    to_timestamp(from_unixtime(col("event_time") / 1000))
)

print(f"Total rows: {all_df.count()}")
all_df.printSchema()
all_df.show(5)

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

fe.create_table(
    name=full_table_name,
    primary_keys=["row_id", "event_time"],
    timestamp_keys=["event_time"],
    df=all_df,
    description="Incremental events feature table with daily ingestion"
)

print(f"Feature table '{full_table_name}' created and data loaded successfully.")

# COMMAND ----------

result = spark.sql(f"SELECT COUNT(*) as cnt FROM {full_table_name}")
result.show()
print("Feature table setup complete.")
