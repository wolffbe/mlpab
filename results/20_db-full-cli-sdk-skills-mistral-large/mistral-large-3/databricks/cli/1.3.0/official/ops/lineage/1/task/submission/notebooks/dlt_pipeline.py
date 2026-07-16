# Databricks notebook source
import dlt
from pyspark.sql.functions import col, round

# Load raw_a.csv into rawa9a7eb8
@dlt.table(
  name="rawa9a7eb8",
  comment="Feature table for raw_a.csv",
  table_properties={
    "quality": "raw"
  }
)
def create_rawa9a7eb8():
    return spark.read.csv("dbfs:/Volumes/workspace/mlpab8a9a8e/staging/raw_a.csv", header=True, inferSchema=True)

# Load raw_b.csv into rawb9a7eb8
@dlt.table(
  name="rawb9a7eb8",
  comment="Feature table for raw_b.csv",
  table_properties={
    "quality": "raw"
  }
)
def create_rawb9a7eb8():
    return spark.read.csv("dbfs:/Volumes/workspace/mlpab8a9a8e/staging/raw_b.csv", header=True, inferSchema=True)

# Create derived table with lineage
@dlt.table(
  name="derived9a7eb8",
  comment="Derived feature table with col_sum = a_val + b_val",
  table_properties={
    "quality": "derived"
  }
)
def create_derived9a7eb8():
    a = dlt.read("rawa9a7eb8")
    b = dlt.read("rawb9a7eb8")
    return (
        a.join(b, "row_id", "inner")
        .select(
            col("row_id"),
            round(col("a_val") + col("b_val"), 6).alias("col_sum")
        )
    )

# Enable online access for derived9a7eb8
@dlt.table(
  name="derived9a7eb8_rt",
  comment="Online table for derived9a7eb8"
)
def create_derived9a7eb8_rt():
    return dlt.read("derived9a7eb8")