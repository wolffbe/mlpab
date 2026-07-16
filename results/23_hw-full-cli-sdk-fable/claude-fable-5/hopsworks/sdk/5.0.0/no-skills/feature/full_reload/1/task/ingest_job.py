"""Runs ON the Hopsworks cluster as a PySpark job: ingests customers v1 and v2."""

import hopsworks
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

project = hopsworks.login()
fs = project.get_feature_store()
spark = SparkSession.builder.getOrCreate()

base = "hdfs:///Projects/{}/Resources".format(project.name)

# --- Version 1: initial export (old schema) ---
df1 = spark.read.csv(base + "/initial_export.csv", header=True, inferSchema=True)
df1 = df1.select(
    col("row_id").cast("string"),
    col("name").cast("string"),
    col("balance_eur").cast("double"),
    col("updated_at").cast("bigint"),
)
print("v1 rows:", df1.count(), "cols:", df1.columns)

fg1 = fs.get_or_create_feature_group(
    name="customers4baff7",
    version=1,
    description="Customers table, initial export (original schema)",
    primary_key=["row_id"],
    event_time="updated_at",
    online_enabled=False,
)
fg1.insert(df1, wait=True)
print("v1 inserted, readback count:", fg1.read().count())

# --- Version 2: full reload with the new, breaking schema ---
df2 = spark.read.csv(base + "/new_export.csv", header=True, inferSchema=True)
df2 = df2.select(
    col("row_id").cast("string"),
    col("full_name").cast("string"),
    col("balance").cast("double"),
    col("currency").cast("string"),
    col("updated_at").cast("bigint"),
)
print("v2 rows:", df2.count(), "cols:", df2.columns)

# re-create v2 from scratch so no stale rows/columns can remain
try:
    existing = fs.get_feature_group("customers4baff7", version=2)
    if existing is not None:
        existing.delete()
        print("deleted pre-existing v2")
except Exception as e:
    print("no pre-existing v2:", type(e).__name__)

fg2 = fs.create_feature_group(
    name="customers4baff7",
    version=2,
    description="Customers table, full re-export (new schema), online-enabled",
    primary_key=["row_id"],
    event_time="updated_at",
    online_enabled=True,
)
fg2.insert(df2, wait=True)

back = fg2.read()
print("v2 offline readback count:", back.count(), "cols:", back.columns)
print("JOB DONE")
