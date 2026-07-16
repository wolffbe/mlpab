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

# Discover all increment files in the volume
files = dbutils.fs.ls(volume_path)
increment_paths = [f.path for f in files if f.name.startswith("increment_") and f.name.endswith(".csv")]
print(f"Found {len(increment_paths)} increment files")

# COMMAND ----------

if increment_paths:
    dfs = [spark.read.csv(p, header=True, schema=schema_def) for p in increment_paths]
    from functools import reduce
    all_df = reduce(lambda a, b: a.union(b), dfs)

    # Convert epoch milliseconds to TIMESTAMP for feature store
    all_df = all_df.withColumn(
        "event_time",
        to_timestamp(from_unixtime(col("event_time") / 1000))
    )

    from databricks.feature_engineering import FeatureEngineeringClient
    fe = FeatureEngineeringClient()

    fe.write_table(
        name=full_table_name,
        df=all_df,
        mode="merge"
    )
    total = all_df.count()
    print(f"Merged {total} rows into {full_table_name}")
    dbutils.notebook.exit(f"Ingested {total} rows from {len(increment_paths)} files")
else:
    print("No increment files found.")
    dbutils.notebook.exit("No increment files found")
