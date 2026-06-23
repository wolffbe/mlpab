#!/usr/bin/env python3
"""
Load item embeddings into Hopsworks vector store and retrieve top-5 nearest neighbors for each query.
"""

import hopsworks
import pandas as pd
import json

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Load data
items_df = pd.read_csv("data/items.csv")
queries_df = pd.read_csv("data/queries.csv")

# Parse embeddings from JSON strings to lists of floats
items_df["embedding"] = items_df["embedding"].apply(lambda x: json.loads(x))
queries_df["embedding"] = queries_df["embedding"].apply(lambda x: json.loads(x))

# Create or get the vector store (feature group)
vector_store_name = "itemsedc1f4"

try:
    fg = fs.get_feature_group(vector_store_name, version=1)
except:
    # Create a new feature group for the vector store (no embedding_index upfront)
    fg = fs.create_feature_group(
        name=vector_store_name,
        version=1,
        description="Vector store for item embeddings",
        primary_key=["item_id"],
        features=[
            {"name": "item_id", "type": "string"},
            {"name": "vector", "type": "array<float>"},
        ],
        online_enabled=True,
    )

# Insert item embeddings into the vector store
items_data = items_df[["item_id", "embedding"]].rename(columns={"embedding": "vector"})
fg.insert(items_data, write_options={"wait_for_job": True})

# Create the vector index after insertion
try:
    vector_index = fs.get_vector_index(vector_store_name, version=1)
except:
    vector_index = fs.create_vector_index(
        name=vector_store_name,
        feature_group=fg,
        vector_col="vector",
        metric="l2_distance",
    )

# Retrieve top-5 nearest neighbors for each query
neighbors = {}
for _, row in queries_df.iterrows():
    query_id = row["query_id"]
    query_embedding = row["embedding"]
    
    # Perform similarity search
    results = vector_index.find_neighbors(
        query_embedding,
        k=5,
        metric="l2_distance",
    )
    
    # Extract item_ids from results
    item_ids = [result["item_id"] for result in results]
    neighbors[query_id] = item_ids

# Write the results to submission/answers.json
with open("submission/answers.json", "w") as f:
    json.dump({"store": vector_store_name, "neighbors": neighbors}, f, indent=2)

print("Vector search completed and results written to submission/answers.json.")