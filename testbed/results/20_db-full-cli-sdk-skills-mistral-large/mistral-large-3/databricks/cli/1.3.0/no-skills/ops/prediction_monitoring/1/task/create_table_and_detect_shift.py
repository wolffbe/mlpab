from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp
from pyspark.sql.types import DoubleType
import pandas as pd
from scipy.stats import ks_2samp

# Initialize Spark
spark = SparkSession.builder.getOrCreate()

# Read the CSV from the volume
df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("dbfs:/Volumes/workspace/mlpabbc4768/mlpabbc4768_volume/prediction_log.csv")

# Convert ts to timestamp and prediction to double
df = df.withColumn("ts", to_timestamp(col("ts"))).withColumn("prediction", col("prediction").cast(DoubleType()))

# Write to Unity Catalog
df.write.saveAsTable("workspace.mlpabbc4768.mlpabbc4768_prediction_log")

# Convert to Pandas for analysis
pdf = df.select("ts", "prediction").toPandas()
pdf = pdf.sort_values("ts")

# Detect shift using Kolmogorov-Smirnov test
window_size = 150  # ~1 day of data
shift_date = None

for i in range(window_size, len(pdf), window_size):
    before = pdf.iloc[i-window_size:i]["prediction"]
    after = pdf.iloc[i:i+window_size]["prediction"]
    ks_stat, p_value = ks_2samp(before, after)
    if p_value < 0.01:  # Significant shift
        shift_date = pdf.iloc[i]["ts"].strftime("%Y-%m-%d")
        break

# Write the result to submission/answers.json
import json
import os
os.makedirs("submission", exist_ok=True)
with open("submission/answers.json", "w") as f:
    json.dump({"onset": shift_date}, f)