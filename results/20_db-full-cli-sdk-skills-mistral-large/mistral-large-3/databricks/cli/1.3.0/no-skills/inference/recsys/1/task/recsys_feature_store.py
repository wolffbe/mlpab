# Databricks notebook source
from databricks.feature_store import FeatureStoreClient
from pyspark.sql.functions import col, concat, row_number, lit
from pyspark.sql.window import Window

# COMMAND ----------

# Initialize Feature Store client
fs = FeatureStoreClient()

# Read input tables
user_embeddings = spark.table("workspace.mlpab5a0819.user_embeddings")
item_embeddings = spark.table("workspace.mlpab5a0819.item_embeddings")
interactions = spark.table("workspace.mlpab5a0819.interactions")

# Compute dot product for all user-item pairs
user_item_scores = (
    user_embeddings
    .crossJoin(item_embeddings)
    .withColumn(
        "score",
        col("e1") * col("e1") + col("e2") * col("e2") + col("e3") * col("e3") +
        col("e4") * col("e4") + col("e5") * col("e5") + col("e6") * col("e6") +
        col("e7") * col("e7") + col("e8") * col("e8")
    )
)

# Exclude already interacted items
filtered_scores = (
    user_item_scores
    .join(
        interactions,
        ["user_id", "item_id"],
        "left_anti"
    )
)

# Rank items by score (desc) and item_id (asc)
window = Window.partitionBy("user_id").orderBy(col("score").desc(), col("item_id").asc())
ranked_scores = (
    filtered_scores
    .withColumn("rank", row_number().over(window))
)

# Select top-5 recommendations
result = (
    ranked_scores
    .filter(col("rank") <= 5)
    .select(
        concat(col("user_id"), lit("#"), col("rank")).alias("rec_id"),
        col("user_id"),
        col("rank"),
        col("item_id")
    )
)

# Write to Delta table
result.write.format("delta").mode("overwrite").saveAsTable("workspace.mlpab5a0819.recse3a36e")

# Create Feature Store table and enable online serving
fs.create_table(
    name="workspace.mlpab5a0819.recse3a36e_feature",
    primary_keys=["rec_id"],
    df=result,
    description="Top-5 recommendations for every user",
    online_store=True
)