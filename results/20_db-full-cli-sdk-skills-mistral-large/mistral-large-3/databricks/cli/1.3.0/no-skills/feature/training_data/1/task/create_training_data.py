# Databricks notebook source
# MAGIC %md
# MAGIC ## Create Training Dataset: churntrainingaf8b21 v1
# MAGIC 
# MAGIC For each `(account_id, label_time)` in `labels.csv`, fetch the most recent feature values from all source tables at or before `label_time`.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Read raw data from volume
volume_path = f"/Volumes/workspace/{spark.conf.get('spark.databricks.clusterUsageTags.schemaName')}/raw_data"

labels_df = spark.read.csv(f"{volume_path}/labels.csv", header=True, inferSchema=True)
transactions_df = spark.read.csv(f"{volume_path}/transactions.csv", header=True, inferSchema=True)
transactions_late_df = spark.read.csv(f"{volume_path}/transactions_late.csv", header=True, inferSchema=True)
profiles_df = spark.read.csv(f"{volume_path}/profiles.csv", header=True, inferSchema=True)
activity_df = spark.read.csv(f"{volume_path}/activity.csv", header=True, inferSchema=True)
health_df = spark.read.csv(f"{volume_path}/account_health.csv", header=True, inferSchema=True)

# Union transactions and transactions_late
transactions_all_df = transactions_df.union(transactions_late_df)

# For each table, filter rows where event_time <= label_time and get the most recent row per account_id
window_spec = Window.partitionBy("account_id").orderBy(F.col("event_time").desc())

# Join all features to labels
result_df = labels_df.alias("labels").join(
    transactions_all_df.alias("t"),
    (F.col("labels.account_id") == F.col("t.account_id")) &
    (F.col("t.event_time") <= F.col("labels.label_time")),
    "left"
).join(
    profiles_df.alias("p"),
    (F.col("labels.account_id") == F.col("p.account_id")) &
    (F.col("p.event_time") <= F.col("labels.label_time")),
    "left"
).join(
    activity_df.alias("a"),
    (F.col("labels.account_id") == F.col("a.account_id")) &
    (F.col("a.event_time") <= F.col("labels.label_time")),
    "left"
).join(
    health_df.alias("h"),
    (F.col("labels.account_id") == F.col("h.account_id")) &
    (F.col("h.event_time") <= F.col("labels.label_time")),
    "left"
)

# For each feature table, select the most recent row per account_id at or before label_time
result_df = result_df.withColumn("t_rank", F.rank().over(window_spec))
result_df = result_df.withColumn("p_rank", F.rank().over(Window.partitionBy("p.account_id").orderBy(F.col("p.event_time").desc())))
result_df = result_df.withColumn("a_rank", F.rank().over(Window.partitionBy("a.account_id").orderBy(F.col("a.event_time").desc())))
result_df = result_df.withColumn("h_rank", F.rank().over(Window.partitionBy("h.account_id").orderBy(F.col("h.event_time").desc())))

result_df = result_df.filter(
    (F.col("t_rank") == 1) &
    (F.col("p_rank") == 1) &
    (F.col("a_rank") == 1) &
    (F.col("h_rank") == 1)
)

# Select the required columns
result_df = result_df.select(
    F.col("labels.account_id"),
    F.col("labels.label_time"),
    F.col("t.amount"),
    F.col("t.balance"),
    F.col("p.credit_score"),
    F.col("p.tier"),
    F.col("a.sessions_7d"),
    F.col("h.health_score"),
    F.col("labels.churned")
)

# Write to Delta table in the isolated schema
output_table = f"{spark.conf.get('spark.databricks.clusterUsageTags.schemaName')}.churntrainingaf8b21"
result_df.write.format("delta").mode("overwrite").saveAsTable(output_table)

# COMMAND ----------

print(f"Training dataset written to {output_table}")