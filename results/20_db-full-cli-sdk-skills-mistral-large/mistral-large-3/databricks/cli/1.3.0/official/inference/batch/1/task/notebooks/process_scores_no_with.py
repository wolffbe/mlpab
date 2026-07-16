# Databricks notebook source
# MAGIC %md
# MAGIC ## Batch Scoring Notebook (No WITH Clauses)

# COMMAND ----------

# Load inputs
T = 1773234000000
feature_history_path = "/Volumes/workspace/mlpab203bd0/mlpab203bd0_volume/feature_history.csv"
model_path = "/Volumes/workspace/mlpab203bd0/mlpab203bd0_volume/model.json"

# Read data
feature_history = spark.read.csv(feature_history_path, header=True, inferSchema=True)
import json
model = json.loads(spark.read.text(model_path).collect()[0]["value"])

# Filter feature history to retain only the most recent revision at or before T
from pyspark.sql import functions as F
from pyspark.sql.window import Window

window_spec = Window.partitionBy("account_id").orderBy(F.col("event_time").desc())
filtered = feature_history.filter(F.col("event_time") <= T)
filtered = filtered.withColumn("rank", F.row_number().over(window_spec))
filtered = filtered.filter(F.col("rank") == 1).drop("rank")

# Compute scores
weights = model["weights"]
bias = model["bias"]

filtered = filtered.withColumn(
    "score",
    F.round(1 / (1 + F.exp(-(
        weights["f1"] * F.col("f1") +
        weights["f2"] * F.col("f2") +
        weights["f3"] * F.col("f3") +
        bias
    ))), 6)
)

# Save results
result = filtered.select("account_id", "score")
result.write.format("delta").mode("overwrite").saveAsTable("workspace.mlpab203bd0.scores4a1a3b")

# Enable online access for low-latency lookup
spark.sql("CREATE TABLE IF NOT EXISTS workspace.mlpab203bd0.scores4a1a3b ONLINE VERSION 1")