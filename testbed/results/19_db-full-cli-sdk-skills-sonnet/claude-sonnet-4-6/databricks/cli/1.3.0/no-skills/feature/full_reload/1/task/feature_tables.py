# Databricks notebook source
# MAGIC %pip install databricks-feature-engineering

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql.functions import col, from_unixtime, to_timestamp

TABLE_NAME = "workspace.mlpab2c4304.customersc31b07"
VOLUME_PATH = "/Volumes/workspace/mlpab2c4304/csvdata"

fe = FeatureEngineeringClient()

def epoch_ms_to_ts(df):
    """Convert updated_at from epoch-ms bigint to TIMESTAMP."""
    return df.withColumn("updated_at", to_timestamp(from_unixtime(col("updated_at").cast("long") / 1000)))

# ── V1: initial schema ────────────────────────────────────────────────────────
print("=== Creating Feature Table V1 ===")

initial_df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(f"{VOLUME_PATH}/initial_export.csv")
)
initial_df = epoch_ms_to_ts(
    initial_df.withColumn("balance_eur", col("balance_eur").cast("double"))
)
initial_df.printSchema()
print(f"V1 row count: {initial_df.count()}")

spark.sql(f"DROP TABLE IF EXISTS {TABLE_NAME}")

fe.create_table(
    name=TABLE_NAME,
    primary_keys=["row_id", "updated_at"],
    timeseries_column="updated_at",
    df=initial_df,
    description="customersc31b07 version 1 – initial schema",
)
print("V1 created.")

# COMMAND ----------

# ── V2: new schema, full reload ───────────────────────────────────────────────
print("=== Creating Feature Table V2 ===")

new_df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(f"{VOLUME_PATH}/new_export.csv")
)
new_df = epoch_ms_to_ts(
    new_df.withColumn("balance", col("balance").cast("double"))
)
new_df.printSchema()
print(f"V2 row count: {new_df.count()}")

# Full drop-and-recreate so no stale v1 rows or columns remain
spark.sql(f"DROP TABLE IF EXISTS {TABLE_NAME}")

fe.create_table(
    name=TABLE_NAME,
    primary_keys=["row_id", "updated_at"],
    timeseries_column="updated_at",
    df=new_df,
    description="customersc31b07 version 2 – new schema (full_name, balance, currency)",
)
print("V2 created.")

# COMMAND ----------

result = spark.sql(f"SELECT * FROM {TABLE_NAME} LIMIT 5")
result.show()
print(f"Final table columns: {result.columns}")
print("Done.")
