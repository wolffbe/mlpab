#!/usr/bin/env python3

import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

# Initialize Spark session
spark = SparkSession.builder.appName("Recommendations").getOrCreate()

# Define schema for input files
user_schema = StructType([
    StructField("user_id", StringType(), False),
    StructField("e1", DoubleType(), True),
    StructField("e2", DoubleType(), True),
    StructField("e3", DoubleType(), True),
    StructField("e4", DoubleType(), True),
    StructField("e5", DoubleType(), True),
    StructField("e6", DoubleType(), True),
    StructField("e7", DoubleType(), True),
    StructField("e8", DoubleType(), True)
])

item_schema = StructType([
    StructField("item_id", StringType(), False),
    StructField("e1", DoubleType(), True),
    StructField("e2", DoubleType(), True),
    StructField("e3", DoubleType(), True),
    StructField("e4", DoubleType(), True),
    StructField("e5", DoubleType(), True),
    StructField("e6", DoubleType(), True),
    StructField("e7", DoubleType(), True),
    StructField("e8", DoubleType(), True)
])

interactions_schema = StructType([
    StructField("user_id", StringType(), False),
    StructField("item_id", StringType(), False)
])

# Read input files
user_embeddings_df = spark.read.csv(
    f"file:{os.getcwd()}/data/user_embeddings.csv", 
    header=True, 
    schema=user_schema
)
item_embeddings_df = spark.read.csv(
    f"file:{os.getcwd()}/data/item_embeddings.csv", 
    header=True, 
    schema=item_schema
)
interactions_df = spark.read.csv(
    f"file:{os.getcwd()}/data/interactions.csv", 
    header=True, 
    schema=interactions_schema
)

# Register temporary views
user_embeddings_df.createOrReplaceTempView("user_embeddings")
item_embeddings_df.createOrReplaceTempView("item_embeddings")
interactions_df.createOrReplaceTempView("interactions")

# Compute dot product between user and item embeddings
recommendations_df = spark.sql("""
    SELECT
        u.user_id,
        i.item_id,
        (u.e1 * i.e1 + u.e2 * i.e2 + u.e3 * i.e3 + u.e4 * i.e4 + 
         u.e5 * i.e5 + u.e6 * i.e6 + u.e7 * i.e7 + u.e8 * i.e8) AS score
    FROM user_embeddings u
    CROSS JOIN item_embeddings i
    WHERE i.item_id NOT IN (
        SELECT item_id FROM interactions WHERE user_id = u.user_id
    )
""")

# Rank items by score (descending) and item_id (ascending for ties)
window_spec = Window.partitionBy("user_id").orderBy(F.desc("score"), F.asc("item_id"))
ranked_recommendations_df = recommendations_df.withColumn("rank", F.row_number().over(window_spec))

# Filter top-5 recommendations per user
top_5_recommendations_df = ranked_recommendations_df.filter(F.col("rank") <= 5)

# Format the output as specified
output_df = top_5_recommendations_df.select(
    F.concat(F.col("user_id"), F.lit("#"), F.col("rank")).alias("rec_id"),
    F.col("user_id"),
    F.col("rank").cast("int"),
    F.col("item_id")
)

# Write to feature table
schema = os.environ["MLPAB_DATABRICKS_SCHEMA"]
feature_table_name = "recse3a36e"

output_df.write.mode("overwrite").saveAsTable(f"{schema}.{feature_table_name}", format="delta")

# Enable low-latency lookup for the feature table
spark.sql(f"""
    CREATE OR REFRESH LIVE TABLE {schema}.{feature_table_name}_online
    AS SELECT * FROM {schema}.{feature_table_name}
""")

# Verify the feature table
print(f"Feature table {schema}.{feature_table_name} created successfully.")
spark.sql(f"SELECT * FROM {schema}.{feature_table_name} LIMIT 10").show()

# Stop Spark session
spark.stop()