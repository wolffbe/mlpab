# Databricks notebook source
import pandas as pd
import numpy as np

# Read input files
requests_df = pd.read_csv("/dbfs/FileStore/requests.csv")
profiles_df = pd.read_csv("/dbfs/FileStore/profiles.csv")

# Merge requests and profiles on account_id
merged_df = pd.merge(requests_df, profiles_df, on="account_id", how="left")

# Compute distance_deg
def calculate_distance_deg(row):
    lat_diff = row["request_lat"] - row["home_lat"]
    lon_diff = row["request_lon"] - row["home_lon"]
    distance = np.sqrt(lat_diff**2 + lon_diff**2)
    return round(distance, 6)

merged_df["distance_deg"] = merged_df.apply(calculate_distance_deg, axis=1)

# Compute score
merged_df["score"] = merged_df.apply(
    lambda row: round(row["base_score"] - 0.1 * row["distance_deg"], 6), axis=1
)

# Select required columns
result_df = merged_df[["request_id", "account_id", "distance_deg", "score"]]

# Write to Delta table
schema_name = os.getenv("MLPAB_DATABRICKS_SCHEMA")
table_name = f"{schema_name}.scoreda4f6e2"
spark.createDataFrame(result_df).write.saveAsTable(
    name=table_name,
    mode="overwrite",
    format="delta"
)

# Enable online table for low-latency lookup
online_table_name = f"{schema_name}.scoreda4f6e2_online"
spark.sql(f"""
CREATE OR REFRESH LIVE TABLE {online_table_name}
AS SELECT * FROM {table_name}
""")

print("Feature table and online table created successfully.")