# Databricks notebook source
import dlt
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType

# Define the schema for the events table
schema = StructType([
    StructField("row_id", StringType(), False),
    StructField("account_id", StringType(), True),
    StructField("event_time", LongType(), False),
    StructField("amount", DoubleType(), True),
    StructField("category", StringType(), True)
])

# Read all increment files from the volume
@dlt.table(
    name="incrementala59b19",
    comment="Feature table for incremental load task",
    table_properties={
        "quality": "gold",
        "pipelines.autoOptimize.managed": "true"
    }
)
def create_events_table():
    # Read all CSV files from the staging volume
    df = spark.read.format("csv") \
        .option("header", "true") \
        .option("inferSchema", "false") \
        .schema(schema) \
        .load("/Volumes/workspace/mlpab620dd6/staging/increment_*.csv")
    
    return df

# Enable online table for low-latency lookup
dlt.create_streaming_live_table(
    name="incrementala59b19_online",
    comment="Online table for incremental load task",
    table_properties={
        "quality": "gold",
        "pipelines.autoOptimize.managed": "true"
    }
)

dlt.apply_changes(
    target="incrementala59b19_online",
    source="incrementala59b19",
    keys=["row_id"],
    sequence_by="event_time",
    apply_as_deletes=None,
    except_column_list=None,
    stored_as_scd_type=2
)