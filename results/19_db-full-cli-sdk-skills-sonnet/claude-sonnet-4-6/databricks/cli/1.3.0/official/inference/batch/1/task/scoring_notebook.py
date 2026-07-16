# Databricks notebook source
# COMMAND ----------
import math

T = 1773306000000

w_f1 = -0.9682
w_f2 = -0.0299
w_f3 = 1.2708
bias = -0.1715

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

# COMMAND ----------
from pyspark.sql import functions as F
from pyspark.sql.window import Window

df = spark.read.csv(
    "/Volumes/workspace/mlpab6ef9cb/data/feature_history.csv",
    header=True,
    inferSchema=True
)

# Filter to events at or before T
df_filtered = df.filter(F.col("event_time") <= T)

# For each account, get the most recent revision at or before T
w = Window.partitionBy("account_id").orderBy(F.col("event_time").desc())
df_latest = (
    df_filtered
    .withColumn("rn", F.row_number().over(w))
    .filter(F.col("rn") == 1)
    .drop("rn", "event_time")
)

# Compute score
df_scored = df_latest.withColumn(
    "score",
    F.round(
        F.lit(1.0) / (F.lit(1.0) + F.exp(-(
            F.lit(w_f1) * F.col("f1") +
            F.lit(w_f2) * F.col("f2") +
            F.lit(w_f3) * F.col("f3") +
            F.lit(bias)
        ))),
        6
    )
).select("account_id", "score")

# COMMAND ----------
# Write feature table as Delta table
df_scored.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("workspace.mlpab6ef9cb.scores4f5893")

print("Feature table created successfully")
df_scored.show(5)
