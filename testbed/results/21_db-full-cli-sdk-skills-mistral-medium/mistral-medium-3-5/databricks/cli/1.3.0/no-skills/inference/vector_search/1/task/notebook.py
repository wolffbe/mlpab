# Databricks notebook source
# MAGIC %md
# MAGIC ## Vector Search Setup and Query

# COMMAND ----------

# Read the items CSV from workspace
items_df = spark.read.csv("/Workspace/Users/benedict@hopsworks.ai/mlpab09d23e/items.csv", header=True, inferSchema=True)

# Display schema
print(items_df.schema)
print(f"Number of items: {items_df.count()}")

# COMMAND ----------

# Create a managed table in Unity Catalog
items_df.write.saveAsTable("workspace.mlpab09d23e.itemsffc8a7")

# COMMAND ----------

# Now create a DELTA_SYNC vector search index
from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()

# Create the index
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

# Wait for the index to be ready
import time
index = vsc.get_index("workspace.mlpab09d23e.itemsffc8a7")
while index.get("status", {}).get("state") != "ONLINE":
    print(f"Index status: {index.get('status', {}).get('state')}")
    time.sleep(10)
    index = vsc.get_index("workspace.mlpab09d23e.itemsffc8a7")

print("Index is ONLINE!")

# COMMAND ----------

# Read queries
queries_df = spark.read.csv("/Workspace/Users/benedict@hopsworks.ai/mlpab09d23e/queries.csv", header=True, inferSchema=True)

# COMMAND ----------

# Query each query vector
import json

results = {}
for row in queries_df.collect():
    query_id = row.query_id
    query_embedding = row.embedding
    
    # Query the index
    search_results = vsc.search(
        index_name="workspace.mlpab09d23e.itemsffc8a7",
        query_vector=query_embedding,
        num_results=5,
        metric_type="L2"
    )
    
    # Extract item_ids from results
    item_ids = [result.get("item_id") for result in search_results.get("results", [])[:5]]
    results[query_id] = item_ids

# COMMAND ----------

# Save results
import json
output = {
    "store": "workspace.mlpab09d23e.itemsffc8a7",
    "neighbors": results
}

with open("/Workspace/Users/benedict@hopsworks.ai/mlpab09d23e/answers.json", "w") as f:
    json.dump(output, f)

print("Results saved!")
