# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType
import math

print("Testing UDF...")

volume_path = "/Volumes/workspace/mlpabfcf9c1/ccfraud_data/"
transactions_path = f"{volume_path}transactions.csv"

df = spark.read.csv(transactions_path, header=True, inferSchema=True)
df = df.withColumn("datetime", F.to_timestamp("datetime", "yyyy-MM-dd'T'HH:mm:ss'Z'"))

print(f"Initial count: {df.count()}")

# Define UDF
def haversine_udf(lat1, lon1, lat2, lon2):
    """Calculate haversine distance between two points in km"""
    try:
        R = 6371.0
        lat1_rad = math.radians(float(lat1)) if lat1 is not None else 0.0
        lon1_rad = math.radians(float(lon1)) if lon1 is not None else 0.0
        lat2_rad = math.radians(float(lat2)) if lat2 is not None else 0.0
        lon2_rad = math.radians(float(lon2)) if lon2 is not None else 0.0
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a)) if a < 1 else 0.0
        return R * c
    except:
        return 0.0

haversine = F.udf(haversine_udf, DoubleType())

# Add card location features
window_spec = Window.partitionBy("cc_num")
df = df.withColumn("card_lat_mean", F.mean("lat").over(window_spec))
df = df.withColumn("card_long_mean", F.mean("long").over(window_spec))

# Apply UDF
df = df.withColumn("distance_from_usual_km", 
    haversine(F.col("lat"), F.col("long"), F.col("card_lat_mean"), F.col("card_long_mean")))

print(f"After UDF: {df.count()}")

# Write to table
df.write.mode("overwrite").saveAsTable("workspace.mlpabfcf9c1.test_udf")
print("UDF table created")
