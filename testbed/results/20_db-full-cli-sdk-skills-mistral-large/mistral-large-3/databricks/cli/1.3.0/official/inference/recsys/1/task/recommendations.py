# Databricks notebook source
# MAGIC %md
# Compute Top-5 Recommendations for Every User

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
import os

# Define schema for the output table
schema = StructType([
    StructField("rec_id", StringType(), False),
    StructField("user_id", StringType(), False),
    StructField("rank", IntegerType(), False),
    StructField("item_id", StringType(), False)
])

# Load input data
interactions_df = spark.read.csv("file:/databricks/driver/data/interactions.csv", header=True, inferSchema=True)
user_embeddings_df = spark.read.csv("file:/databricks/driver/data/user_embeddings.csv", header=True, inferSchema=True)
item_embeddings_df = spark.read.csv("file:/databricks/driver/data/item_embeddings.csv", header=True, inferSchema=True)

# COMMAND ----------

# Extract embedding columns for users and items
embedding_cols = ["e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8"]

# Compute dot product between user and item embeddings
# Cross join users and items, then compute dot product
user_item_df = user_embeddings_df.crossJoin(item_embeddings_df)

for col in embedding_cols:
    user_item_df = user_item_df.withColumn(f"dot_{col}", F.col(f"user_embeddings_df.{col}") * F.col(f"item_embeddings_df.{col}"))

# Sum the dot products to get the relevance score
relevance_df = user_item_df.withColumn(
    "relevance",
    sum(F.col(f"dot_{col}") for col in embedding_cols)
)

# Drop intermediate columns
for col in embedding_cols:
    relevance_df = relevance_df.drop(f"dot_{col}")

# Select relevant columns
relevance_df = relevance_df.select(
    F.col("user_embeddings_df.user_id").alias("user_id"),
    F.col("item_embeddings_df.item_id").alias("item_id"),
    F.col("relevance")
)

# COMMAND ----------

# Exclude items the user has already interacted with
excluded_interactions_df = relevance_df.join(
    interactions_df,
    ["user_id", "item_id"],
    "left_anti"
)

# COMMAND ----------

# Rank items by relevance, breaking ties by ascending item_id
window_spec = Window.partitionBy("user_id").orderBy(F.desc("relevance"), F.asc("item_id"))
ranked_df = excluded_interactions_df.withColumn("rank", F.row_number().over(window_spec))

# Filter top-5 recommendations per user
top_5_df = ranked_df.filter(F.col("rank") <= 5)

# COMMAND ----------

# Create rec_id column
result_df = top_5_df.withColumn("rec_id", F.concat(F.col("user_id"), F.lit("#"), F.col("rank")))

# Select and order columns as specified
result_df = result_df.select("rec_id", "user_id", "rank", "item_id")

# COMMAND ----------

# Define the schema name from environment variables
schema_name = os.environ.get("MLPAB_DATABRICKS_SCHEMA")

# Write the result to a feature table
result_df.write.saveAsTable(f"{schema_name}.recse3a36e", mode="overwrite")

# Enable online access for the feature table
spark.sql(f"CREATE TABLE IF NOT EXISTS {schema_name}.recse3a36e ONLINE USING DELTA SHARE (FEATURE_STORE ON)")

# COMMAND ----------

# Verify the table was created
display(spark.sql(f"SELECT * FROM {schema_name}.recse3a36e LIMIT 10"))