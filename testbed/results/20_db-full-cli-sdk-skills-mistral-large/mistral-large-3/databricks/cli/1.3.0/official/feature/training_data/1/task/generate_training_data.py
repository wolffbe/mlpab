# Databricks notebook source
# MAGIC %md
# Generate Training Dataset: churntrainingaf8b21, version 1

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# COMMAND ----------

# Read raw data from the volume
volume_path = f"/Volumes/workspace/mlpab674210/{dbutils.widgets.get("prefix")}_raw_data"

labels_df = spark.read.csv(f"{volume_path}/labels.csv", header=True, inferSchema=True)
transactions_df = spark.read.csv(f"{volume_path}/transactions.csv", header=True, inferSchema=True)
transactions_late_df = spark.read.csv(f"{volume_path}/transactions_late.csv", header=True, inferSchema=True)
profiles_df = spark.read.csv(f"{volume_path}/profiles.csv", header=True, inferSchema=True)
activity_df = spark.read.csv(f"{volume_path}/activity.csv", header=True, inferSchema=True)
account_health_df = spark.read.csv(f"{volume_path}/account_health.csv", header=True, inferSchema=True)

# COMMAND ----------

# Merge transactions and transactions_late
transactions_merged_df = transactions_df.union(transactions_late_df)

# COMMAND ----------

# For each (account_id, label_time), get the most recent feature values at or before label_time
# Define a window for each (account_id, label_time), ordered by event_time descending
window_spec = Window.partitionBy("account_id", "label_time").orderBy(F.col("event_time").desc())

# Cross-join labels with each feature table, filter to event_time <= label_time, then get the most recent row
transactions_with_labels_df = (
    labels_df
    .crossJoin(transactions_merged_df)
    .filter(transactions_merged_df.event_time <= labels_df.label_time)
    .withColumn("rank", F.row_number().over(window_spec))
    .filter(F.col("rank") == 1)
    .drop("rank")
)

profiles_with_labels_df = (
    labels_df
    .crossJoin(profiles_df)
    .filter(profiles_df.event_time <= labels_df.label_time)
    .withColumn("rank", F.row_number().over(window_spec))
    .filter(F.col("rank") == 1)
    .drop("rank")
)

activity_with_labels_df = (
    labels_df
    .crossJoin(activity_df)
    .filter(activity_df.event_time <= labels_df.label_time)
    .withColumn("rank", F.row_number().over(window_spec))
    .filter(F.col("rank") == 1)
    .drop("rank")
)

account_health_with_labels_df = (
    labels_df
    .crossJoin(account_health_df)
    .filter(account_health_df.event_time <= labels_df.label_time)
    .withColumn("rank", F.row_number().over(window_spec))
    .filter(F.col("rank") == 1)
    .drop("rank")
)

# COMMAND ----------

# Join all features to labels_df
joined_df = (
    labels_df
    .join(
        transactions_with_labels_df,
        [labels_df.account_id, labels_df.label_time],
        "left"
    )
    .join(
        profiles_with_labels_df,
        [labels_df.account_id, labels_df.label_time],
        "left"
    )
    .join(
        activity_with_labels_df,
        [labels_df.account_id, labels_df.label_time],
        "left"
    )
    .join(
        account_health_with_labels_df,
        [labels_df.account_id, labels_df.label_time],
        "left"
    )
    .select(
        labels_df.account_id,
        labels_df.label_time,
        transactions_with_labels_df.amount,
        transactions_with_labels_df.balance,
        profiles_with_labels_df.credit_score,
        profiles_with_labels_df.tier,
        activity_with_labels_df.sessions_7d,
        account_health_with_labels_df.health_score,
        labels_df.churned
    )
)

# COMMAND ----------

# Write the result to a Delta table in the Unity Catalog schema
output_table = f"workspace.mlpab674210.churntrainingaf8b21"
joined_df.write.format("delta").mode("overwrite").saveAsTable(output_table)

# Register the table as version 1
display(spark.sql(f"DESCRIBE HISTORY {output_table}"))