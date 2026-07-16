# Create transactions9dd1da feature table
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.getOrCreate()

schema = "workspace.mlpabcbef07"
table_name = f"{schema}.transactions9dd1da"
volume_path = "/Volumes/workspace/mlpabcbef07/csvdata"

# Read both CSV exports
df1 = spark.read.csv(f"{volume_path}/transactions_export_1.csv", header=True, inferSchema=True)
df2 = spark.read.csv(f"{volume_path}/transactions_export_2.csv", header=True, inferSchema=True)

# Union and deduplicate by row_id (files overlap)
df_all = df1.union(df2)
df_dedup = df_all.dropDuplicates(["row_id"])

# Ensure correct types
df_final = df_dedup.select(
    col("row_id").cast("string"),
    col("account_id").cast("string"),
    col("event_time").cast("long"),
    col("amount").cast("double"),
    col("category").cast("string")
)

# Write as Delta table
df_final.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(table_name)

print(f"Table created: {table_name}")
count = spark.table(table_name).count()
print(f"Row count: {count}")

distinct_count = spark.table(table_name).select("row_id").distinct().count()
print(f"Distinct row_ids: {distinct_count}")
