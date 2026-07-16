#!/usr/bin/env python3
"""
Ingest items into Hopsworks vector store and perform similarity search for queries.
"""
import hopsworks
import pandas as pd
import json
import numpy as np
import os

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Load data
items_df = pd.read_csv("data/items.csv")
queries_df = pd.read_csv("data/queries.csv")

# Parse embeddings
items_df["embedding_array"] = items_df["embedding"].apply(lambda x: np.array(json.loads(x), dtype=np.float32))
queries_df["embedding_array"] = queries_df["embedding"].apply(lambda x: np.array(json.loads(x), dtype=np.float32))

# Create vector store
vector_db = project.get_vector_database()
index_name = "itemsedc1f4"

# Check if index exists, delete if it does (for idempotency)
try:
    existing_index = vector_db.get_index(index_name)
    existing_index.delete()
except:
    pass

# Create index
index = vector_db.create_index(
    name=index_name,
    description="Vector store for items",
    embedding_dimension=16,
    similarity_metric="l2_distance"
)

# Ingest items
items = items_df[["item_id", "embedding_array"]].rename(columns={"item_id": "id", "embedding_array": "embedding"})
items["id"] = items["id"].astype(str)
index.insert(items)

# Perform similarity search for all queries
neighbors = {}
for _, row in queries_df.iterrows():
    query_id = row["query_id"]
    query_embedding = row["embedding_array"]
    
    results = index.find_nearest_neighbors(
        query_embedding,
        k=5,
        include_distances=False
    )
    neighbors[query_id] = [str(item_id) for item_id in results["id"]]

# Write results
with open("submission/answers.json", "w") as f:
    json.dump({
        "store": index_name,
        "neighbors": neighbors
    }, f, indent=2)