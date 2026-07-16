# Databricks notebook source
# COMMAND ----------
# Compute top-5 recommendations using dot product similarity

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Read data
user_emb = spark.read.csv("/Volumes/workspace/mlpabb40f43/data/user_embeddings.csv", header=True, inferSchema=True)
item_emb = spark.read.csv("/Volumes/workspace/mlpabb40f43/data/item_embeddings.csv", header=True, inferSchema=True)
interactions = spark.read.csv("/Volumes/workspace/mlpabb40f43/data/interactions.csv", header=True)

# Rename item embedding columns to avoid conflict during cross join
item_emb_renamed = item_emb.select(
    F.col("item_id"),
    F.col("e1").alias("ie1"),
    F.col("e2").alias("ie2"),
    F.col("e3").alias("ie3"),
    F.col("e4").alias("ie4"),
    F.col("e5").alias("ie5"),
    F.col("e6").alias("ie6"),
    F.col("e7").alias("ie7"),
    F.col("e8").alias("ie8"),
)

# Cross join to get all user-item pairs
pairs = user_emb.crossJoin(item_emb_renamed)

# Compute dot product score
pairs_scored = pairs.withColumn(
    "score",
    F.col("e1") * F.col("ie1") +
    F.col("e2") * F.col("ie2") +
    F.col("e3") * F.col("ie3") +
    F.col("e4") * F.col("ie4") +
    F.col("e5") * F.col("ie5") +
    F.col("e6") * F.col("ie6") +
    F.col("e7") * F.col("ie7") +
    F.col("e8") * F.col("ie8")
).select("user_id", "item_id", "score")

# Exclude already-interacted items
pairs_filtered = pairs_scored.join(
    interactions,
    on=["user_id", "item_id"],
    how="left_anti"
)

# Rank by descending score, tie-break by ascending item_id
window_spec = Window.partitionBy("user_id").orderBy(F.col("score").desc(), F.col("item_id").asc())
ranked = pairs_filtered.withColumn("rank", F.row_number().over(window_spec))

# Keep top 5
top5 = ranked.filter(F.col("rank") <= 5)

# Create rec_id in format "<user_id>#<rank>"
result = top5.withColumn(
    "rec_id",
    F.concat(F.col("user_id"), F.lit("#"), F.col("rank").cast("string"))
).select(
    F.col("rec_id"),
    F.col("user_id"),
    F.col("rank").cast("int"),
    F.col("item_id")
)

result.show(20)
print(f"Total rows: {result.count()}")

# COMMAND ----------
# Write to feature table in Unity Catalog

table_name = "workspace.mlpabb40f43.recs708df6"

# Drop if exists and recreate
spark.sql(f"DROP TABLE IF EXISTS {table_name}")

result.write.format("delta").mode("overwrite").saveAsTable(table_name)

# Set the primary key and other table properties for feature store compatibility
spark.sql(f"""
ALTER TABLE {table_name}
SET TBLPROPERTIES (
    'delta.feature.allowColumnDefaults' = 'supported'
)
""")

print(f"Feature table {table_name} created successfully")

# COMMAND ----------
# Verify the table
spark.sql(f"SELECT * FROM {table_name} ORDER BY user_id, rank LIMIT 20").show()
spark.sql(f"SELECT COUNT(*) as total_rows FROM {table_name}").show()
