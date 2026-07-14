# Databricks notebook source
# MAGIC %md
# MAGIC ## Vector Search Setup and Query

# COMMAND ----------

import csv
import json
import requests
import time
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

# Create DELTA_SYNC index using REST API
import os
DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST", "https://***REDACTED***")
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN")

headers = {
    "Authorization": f"Bearer {DATABRICKS_TOKEN}",
    "Content-Type": "application/json"
}

# Create index
index_data = {
    "name": "workspace.mlpab09d23e.itemsffc8a7",
    "endpoint_name": "mlpab09d23e_itemsffc8a7",
    "primary_key": "item_id",
    "index_type": "DELTA_SYNC",
    "index_subtype": "HYBRID",
    "delta_sync_index_spec": {
        "source_table": "workspace.mlpab09d23e.itemsffc8a7",
        "embedding_vector_column": "embedding"
    }
}

response = requests.post(
    f"{DATABRICKS_HOST}/api/2.0/vector-search/indexes",
    headers=headers,
    json=index_data
)
print(f"Create index response: {response.status_code}, {response.text}")

# COMMAND ----------

# Wait for index to be ready
import time
while True:
    response = requests.get(
        f"{DATABRICKS_HOST}/api/2.0/vector-search/indexes?endpoint_name=mlpab09d23e_itemsffc8a7&index_name=workspace.mlpab09d23e.itemsffc8a7",
        headers=headers
    )
    index_info = response.json()
    if index_info.get("status", {}).get("state") == "ONLINE":
        print("Index is ONLINE!")
        break
    print(f"Index status: {index_info.get('status', {}).get('state')}")
    time.sleep(10)

# COMMAND ----------

# Query each query
queries = []
with open("/Workspace/Users/benedict@hopsworks.ai/mlpab09d23e/queries.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        queries.append(row)

results = {}
for query in queries:
    query_id = query["query_id"]
    query_embedding = json.loads(query["embedding"])
    
    # Query the index
    query_data = {
        "index_name": "workspace.mlpab09d23e.itemsffc8a7",
        "query_vector": query_embedding,
        "num_results": 5,
        "metric_type": "L2"
    }
    
    response = requests.post(
        f"{DATABRICKS_HOST}/api/2.0/vector-search/indexes/query",
        headers=headers,
        json=query_data
    )
    
    search_results = response.json()
    item_ids = [result.get("item_id") for result in search_results.get("results", [])[:5]]
    results[query_id] = item_ids
    print(f"Query {query_id}: {item_ids}")

# COMMAND ----------

# Save results
output = {
    "store": "workspace.mlpab09d23e.itemsffc8a7",
    "neighbors": results
}

with open("/Workspace/Users/benedict@hopsworks.ai/mlpab09d23e/answers.json", "w") as f:
    json.dump(output, f)

print("Results saved!")
