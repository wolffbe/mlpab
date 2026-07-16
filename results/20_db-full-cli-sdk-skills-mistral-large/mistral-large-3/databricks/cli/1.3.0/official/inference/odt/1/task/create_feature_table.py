# Databricks notebook source
# MAGIC %md
# MAGIC ## Create Feature Table: scoreda4f6e2
# MAGIC 
# MAGIC This notebook:
# MAGIC 1. Reads the input files from the local path.
# MAGIC 2. Joins them on `account_id`.
# MAGIC 3. Computes `distance_deg` and `score` as specified.
# MAGIC 4. Writes the results to a feature table named `scoreda4f6e2` in the schema `workspace.mlpab3f631d`.
# MAGIC 5. Enables online access for low-latency lookup.

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType

# COMMAND ----------

# Read input files
requests_df = spark.read.csv("/Workspace/mlpab3f631d/requests.csv", header=True, inferSchema=True)
profiles_df = spark.read.csv("/Workspace/mlpab3f631d/profiles.csv", header=True, inferSchema=True)

# COMMAND ----------

# Join requests and profiles on account_id
joined_df = requests_df.join(profiles_df, "account_id", "inner")

# COMMAND ----------

# Compute distance_deg and score
distance_udf = F.udf(
    lambda req_lat, req_lon, home_lat, home_lon: round(((req_lat - home_lat) ** 2 + (req_lon - home_lon) ** 2) ** 0.5, 6), 
    DoubleType()
)

scored_df = joined_df.withColumn(
    "distance_deg", 
    distance_udf(F.col("request_lat"), F.col("request_lon"), F.col("home_lat"), F.col("home_lon"))
).withColumn(
    "score", 
    F.round(F.col("base_score") - 0.1 * F.col("distance_deg"), 6)
).select(
    "request_id", 
    "account_id", 
    "distance_deg", 
    "score"
)

# COMMAND ----------

# Write to feature table
schema = "workspace.mlpab3f631d"
table_name = "scoreda4f6e2"

scored_df.write.saveAsTable(f"{schema}.{table_name}", mode="overwrite")

# COMMAND ----------

# Enable online access for the feature table
from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()
fe.create_table(
    name=f"{schema}.{table_name}_online",
    primary_keys=["request_id"],
    df=scored_df,
    mode="overwrite"
)

# Publish the online table for low-latency lookup
fe.publish_table(f"{schema}.{table_name}_online")

# COMMAND ----------

# Verify the feature table
print(f"Feature table {schema}.{table_name} created successfully.")
spark.sql(f"SELECT * FROM {schema}.{table_name} LIMIT 5").show()