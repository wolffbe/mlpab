# Databricks notebook source
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# Create rawa9a7eb8
spark.sql("""
CREATE EXTERNAL TABLE IF NOT EXISTS workspace.mlpab5e2cc3.rawa9a7eb8 (
  row_id STRING,
  a_val DOUBLE
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES ('separatorChar' = ',', 'quoteChar' = '\"', 'escapeChar' = '\\')
STORED AS TEXTFILE
LOCATION 'dbfs:/Volumes/workspace/mlpab5e2cc3/staging/raw_a.csv'
TBLPROPERTIES ('skip.header.line.count' = '1')
""")

# Create rawb9a7eb8
spark.sql("""
CREATE EXTERNAL TABLE IF NOT EXISTS workspace.mlpab5e2cc3.rawb9a7eb8 (
  row_id STRING,
  b_val DOUBLE
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES ('separatorChar' = ',', 'quoteChar' = '\"', 'escapeChar' = '\\')
STORED AS TEXTFILE
LOCATION 'dbfs:/Volumes/workspace/mlpab5e2cc3/staging/raw_b.csv'
TBLPROPERTIES ('skip.header.line.count' = '1')
""")

# Create derived table
spark.sql("""
CREATE OR REPLACE TABLE workspace.mlpab5e2cc3.derived9a7eb8 AS
SELECT
  a.row_id,
  ROUND(a.a_val + b.b_val, 6) AS col_sum
FROM workspace.mlpab5e2cc3.rawa9a7eb8 a
JOIN workspace.mlpab5e2cc3.rawb9a7eb8 b
ON a.row_id = b.row_id
""")

# Enable online table for derived9a7eb8
spark.sql("CREATE OR REFRESH ONLINE TABLE workspace.mlpab5e2cc3.derived9a7eb8")