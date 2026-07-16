# Databricks notebook source
import dlt
from pyspark.sql.functions import col

# Read the raw events
raw_events = spark.read.csv("file:/Workspace/Repos/${MLPAB_DATABRICKS_PREFIX}/task/data/events.csv", header=True, inferSchema=True)

# Define the contract rules
valid_categories = ["grocery", "travel", "salary", "rent", "other"]

# Filter rows that satisfy the contract
filtered_events = raw_events.filter(
    (col("amount").isNotNull()) &
    (col("amount") >= 0) &
    (col("amount") <= 10000) &
    (col("category").isin(valid_categories))
)

# Register the feature table
@dlt.table(
    name="events45bd4b",
    comment="Filtered events table with only valid rows per contract",
    table_properties={
        "quality": "gold",
        "pipelines.autoOptimize.managed": "true"
    }
)
def create_events_table():
    return filtered_events

# Enable online table for low-latency lookup
dlt.create_streaming_live_table(
    name="events45bd4b_online",
    comment="Online table for low-latency lookup of events45bd4b",
    table_properties={
        "pipelines.autoOptimize.managed": "true"
    }
)

dlt.apply_changes(
    target="events45bd4b_online",
    source="events45bd4b",
    keys=["row_id"],
    sequence_by=col("event_time"),
    apply_as_deletes=None,
    except_column_list=None,
    stored_as_scd_type=2
)