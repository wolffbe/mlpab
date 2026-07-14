# Databricks notebook source
import dlt
from pyspark.sql.functions import col, concat, row_number
from pyspark.sql.window import Window

# COMMAND ----------

@dlt.table(
  name="recse3a36e",
  comment="Top-5 recommendations for every user",
  table_properties={
    "quality": "gold",
    "pipelines.autoOptimize.managed": "true"
  }
)
def create_recse3a36e():
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
    
    return result

# COMMAND ----------

@dlt.table(
  name="recse3a36e_online",
  comment="Online table for low-latency lookup",
  table_properties={
    "quality": "gold",
    "pipelines.autoOptimize.managed": "true",
    "delta.autoOptimize.optimizeWrite": "true",
    "delta.autoOptimize.autoCompact": "true"
  }
)
def create_recse3a36e_online():
    return dlt.read("recse3a36e")