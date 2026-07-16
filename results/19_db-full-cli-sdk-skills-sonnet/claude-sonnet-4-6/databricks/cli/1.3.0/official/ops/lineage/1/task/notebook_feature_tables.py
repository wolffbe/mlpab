# Databricks notebook source

import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

SCHEMA = "workspace.mlpab894195"
VOLUME_PATH = "/Volumes/workspace/mlpab894195/data_vol"

# COMMAND ----------
# Create rawad05474 with NOT NULL primary key
spark.sql(f"DROP TABLE IF EXISTS {SCHEMA}.rawad05474")
spark.sql(f"""
CREATE TABLE {SCHEMA}.rawad05474 (
    row_id STRING NOT NULL,
    a_val DOUBLE
) USING DELTA
COMMENT 'Raw feature table A from raw_a.csv'
""")

# Load raw_a CSV with proper schema and insert
raw_a_df = spark.read.option("header", "true").option("inferSchema", "true").csv(f"{VOLUME_PATH}/raw_a.csv")
raw_a_df = raw_a_df.withColumn("row_id", F.col("row_id").cast(StringType()))
raw_a_df = raw_a_df.withColumn("a_val", F.col("a_val").cast(DoubleType()))
raw_a_df.select("row_id", "a_val").write.format("delta").mode("append").saveAsTable(f"{SCHEMA}.rawad05474")

spark.sql(f"ALTER TABLE {SCHEMA}.rawad05474 ADD CONSTRAINT rawad05474_pk PRIMARY KEY (row_id)")
cnt_a = spark.table(f"{SCHEMA}.rawad05474").count()
print(f"Created {SCHEMA}.rawad05474 with {cnt_a} rows")

# COMMAND ----------
# Create rawbd05474 with NOT NULL primary key
spark.sql(f"DROP TABLE IF EXISTS {SCHEMA}.rawbd05474")
spark.sql(f"""
CREATE TABLE {SCHEMA}.rawbd05474 (
    row_id STRING NOT NULL,
    b_val DOUBLE
) USING DELTA
COMMENT 'Raw feature table B from raw_b.csv'
""")

# Load raw_b CSV with proper schema and insert
raw_b_df = spark.read.option("header", "true").option("inferSchema", "true").csv(f"{VOLUME_PATH}/raw_b.csv")
raw_b_df = raw_b_df.withColumn("row_id", F.col("row_id").cast(StringType()))
raw_b_df = raw_b_df.withColumn("b_val", F.col("b_val").cast(DoubleType()))
raw_b_df.select("row_id", "b_val").write.format("delta").mode("append").saveAsTable(f"{SCHEMA}.rawbd05474")

spark.sql(f"ALTER TABLE {SCHEMA}.rawbd05474 ADD CONSTRAINT rawbd05474_pk PRIMARY KEY (row_id)")
cnt_b = spark.table(f"{SCHEMA}.rawbd05474").count()
print(f"Created {SCHEMA}.rawbd05474 with {cnt_b} rows")

# COMMAND ----------
# Create derived table derivedd05474 with lineage (via SQL INSERT)
spark.sql(f"DROP TABLE IF EXISTS {SCHEMA}.derivedd05474")
spark.sql(f"""
CREATE TABLE {SCHEMA}.derivedd05474 (
    row_id STRING NOT NULL,
    col_sum DOUBLE
) USING DELTA
COMMENT 'Derived from rawad05474 and rawbd05474: col_sum = a_val + b_val'
""")
spark.sql(f"""
INSERT INTO {SCHEMA}.derivedd05474
SELECT
    a.row_id,
    ROUND(a.a_val + b.b_val, 6) AS col_sum
FROM {SCHEMA}.rawad05474 a
INNER JOIN {SCHEMA}.rawbd05474 b ON a.row_id = b.row_id
""")
spark.sql(f"ALTER TABLE {SCHEMA}.derivedd05474 ADD CONSTRAINT derivedd05474_pk PRIMARY KEY (row_id)")
cnt_d = spark.table(f"{SCHEMA}.derivedd05474").count()
print(f"Created {SCHEMA}.derivedd05474 with {cnt_d} rows")

# COMMAND ----------
# Verify results
print("\n=== derivedd05474 preview ===")
spark.table(f"{SCHEMA}.derivedd05474").orderBy("row_id").show(50)

# COMMAND ----------
# Print CSV data for floor submission
result = spark.table(f"{SCHEMA}.derivedd05474").orderBy("row_id").collect()
print("row_id,col_sum")
for row in result:
    print(f"{row['row_id']},{row['col_sum']}")

print("\nAll feature tables created successfully!")
