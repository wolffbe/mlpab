# Databricks notebook source
# MAGIC %md
# MAGIC ## Feature Engineering for Fraud Detection
# MAGIC 
# MAGIC This notebook engineers fraud-specific features from transaction data:
# MAGIC - Transaction velocity (1h, 24h)
# MAGIC - Amount velocity (1h, 24h)
# MAGIC - Geo distance from card's usual location
# MAGIC - Amount z-score

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window
import math

# --- Input/Output ---
input_path = spark.conf.get("input.path")
output_table = spark.conf.get("output.table")

# --- Read Data ---
df = spark.read.csv(input_path, header=True, inferSchema=True)

# --- Feature Engineering ---
# 1. Transaction Velocity (1h, 24h)
window_1h = Window.partitionBy("cc_num").orderBy("datetime").rangeBetween(-3600, 0)
window_24h = Window.partitionBy("cc_num").orderBy("datetime").rangeBetween(-86400, 0)

df = df.withColumn("txn_velocity_1h", F.count("*").over(window_1h) - 1)

df = df.withColumn("txn_velocity_24h", F.count("*").over(window_24h) - 1)

# 2. Amount Velocity (1h, 24h)
df = df.withColumn("amount_velocity_1h", F.sum("amount").over(window_1h) - F.col("amount"))

df = df.withColumn("amount_velocity_24h", F.sum("amount").over(window_24h) - F.col("amount"))

# 3. Geo Distance (Haversine formula)
def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two points on the Earth."""
    R = 6371  # Earth radius in km
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = (math.sin(dLat / 2) * math.sin(dLat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dLon / 2) * math.sin(dLon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

haversine_udf = F.udf(haversine)

# Calculate the card's usual location (median lat/long)
usual_location = df.groupBy("cc_num").agg(
    F.median("lat").alias("usual_lat"),
    F.median("long").alias("usual_long")
)

df = df.join(usual_location, on="cc_num", how="left")

df = df.withColumn("geo_distance_km", haversine_udf(F.col("usual_lat"), F.col("usual_long"), F.col("lat"), F.col("long")))

# 4. Amount Z-Score (per card)
stats = df.groupBy("cc_num").agg(
    F.mean("amount").alias("mean_amount"),
    F.stddev("amount").alias("std_amount")
)

df = df.join(stats, on="cc_num", how="left")

df = df.withColumn("amount_zscore", (F.col("amount") - F.col("mean_amount")) / F.col("std_amount"))

# --- Write to Feature Table ---
(df.write
   .format("delta")
   .mode("overwrite")
   .saveAsTable(output_table))

print(f"Feature table {output_table} updated successfully.")