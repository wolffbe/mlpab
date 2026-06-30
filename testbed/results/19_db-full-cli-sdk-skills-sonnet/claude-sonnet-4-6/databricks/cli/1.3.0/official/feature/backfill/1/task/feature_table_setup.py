# Databricks notebook source
# COMMAND ----------
from pyspark.sql.functions import col, row_number
from pyspark.sql.window import Window
from pyspark.sql.types import StringType, DoubleType, LongType

# COMMAND ----------
volume_path = "/Volumes/workspace/mlpab0442b8/mlpab0442b8_uploads"

df1 = spark.read.csv(f"{volume_path}/batch_1.csv", header=True, inferSchema=True)
df2 = spark.read.csv(f"{volume_path}/batch_2.csv", header=True, inferSchema=True)
df3 = spark.read.csv(f"{volume_path}/batch_3.csv", header=True, inferSchema=True)

all_data = df1.union(df2).union(df3)

window_spec = Window.partitionBy("row_id").orderBy(col("updated_at").desc())
latest_data = (
    all_data
    .withColumn("_rank", row_number().over(window_spec))
    .filter(col("_rank") == 1)
    .drop("_rank")
    .withColumn("row_id", col("row_id").cast(StringType()))
    .withColumn("status", col("status").cast(StringType()))
    .withColumn("balance", col("balance").cast(DoubleType()))
    .withColumn("updated_at", col("updated_at").cast(LongType()))
)

table_name = "workspace.mlpab0442b8.accountse81ff1"

latest_data.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(table_name)

count = spark.table(table_name).count()
print(f"Written {count} rows to {table_name}")

# COMMAND ----------
try:
    from databricks.feature_engineering import FeatureEngineeringClient
    fe = FeatureEngineeringClient()
    fe.register_table(
        name=table_name,
        primary_keys=["row_id"],
        timestamp_keys=["updated_at"],
        description="Accounts feature table - latest revision per row_id"
    )
    print(f"Registered as feature table: {table_name}")
except Exception as e1:
    print(f"FE client error: {e1}")
    try:
        from databricks import feature_store
        fs = feature_store.FeatureStoreClient()
        fs.register_table(
            delta_table=table_name,
            primary_keys=["row_id"],
            timestamp_keys=["updated_at"],
            description="Accounts feature table - latest revision per row_id"
        )
        print(f"Registered via FeatureStoreClient: {table_name}")
    except Exception as e2:
        print(f"FS client error: {e2}")

print("Feature table setup complete")
