# Databricks notebook source
# MAGIC %md
# MAGIC ## Vector Search Setup and Query

# COMMAND ----------

import csv
import json
from databricks.vector_search.client import VectorSearchClient
from pyspark.sql import SparkSession

# Read items
items = []
with open("/Workspace/Users/benedict@hopsworks.ai/mlpab09d23e/items.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        items.append(row)

# Create DataFrame
spark = SparkSession.builder.getOrCreate()
items_df = spark.createDataFrame(items)
items_df.write.saveAsTable("workspace.mlpab09d23e.itemsffc8a7")

# COMMAND ----------

# Create index
vsc = VectorSearchClient()
vsc.create_delta_sync_index(
    name="workspace.mlpab09d23e.itemsffc8a7",
    source_table_name="workspace.mlpab09d23e.itemsffc8a7",
    endpoint_name="mlpab09d23e_itemsffc8a7",
    primary_key="item_id",
    embedding_vector_column="embedding",
    embedding_dimension=16,
    metric_type="L2"
)

# COMMAND ----------

# Wait for index
import time
index = vsc.get_index("workspace.mlpab09d23e.itemsffc8a7")
while index.get("status", {}).get("state") != "ONLINE":
    print(f"Index status: {index.get('status', {}).get('state')}")
    time.sleep(10)
    index = vsc.get_index("workspace.mlpab09d23e.itemsffc8a7")

print("Index is ONLINE!")

# COMMAND ----------

# Query
queries = []
with open("/Workspace/Users/benedict@hopsworks.ai/mlpab09d23e/queries.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        queries.append(row)

results = {}
for query in queries:
    query_id = query["query_id"]
    query_embedding = json.loads(query["embedding"])
    search_results = vsc.search(
        index_name="workspace.mlpab09d23e.itemsffc8a7",
        query_vector=query_embedding,
        num_results=5,
        metric_type="L2"
    )
    item_ids = [result.get("item_id") for result in search_results.get("results", [])[:5]]
    results[query_id] = item_ids

# COMMAND ----------

# Save
output = {"store": "workspace.mlpab09d23e.itemsffc8a7", "neighbors": results}
with open("/Workspace/Users/benedict@hopsworks.ai/mlpab09d23e/answers.json", "w") as f:
    json.dump(output, f)

print("Results saved!")
