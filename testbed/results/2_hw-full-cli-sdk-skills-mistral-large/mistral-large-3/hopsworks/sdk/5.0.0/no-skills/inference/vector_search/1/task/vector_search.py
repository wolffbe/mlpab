#!/usr/bin/env python3

import hopsworks
import pandas as pd
import json
import os
from hsfs import embedding

# Load data
items_df = pd.read_csv("data/items.csv")
queries_df = pd.read_csv("data/queries.csv")

# Parse embeddings from JSON strings to lists of floats
items_df["embedding"] = items_df["embedding"].apply(json.loads)
queries_df["embedding"] = queries_df["embedding"].apply(json.loads)

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Create or get the feature group with embedding index
embedding_index = embedding.EmbeddingIndex(index_name="itemsedc1f4")
embedding_index.add_embedding(name="embedding", dimension=16, similarity_function_type="l2_norm")

try:
    fg = fs.get_feature_group(name="itemsedc1f4", version=1)
except:
    fg = fs.create_feature_group(
        name="itemsedc1f4",
        version=1,
        primary_key=["item_id"],
        embedding_index=embedding_index,
        online_enabled=True,
    )
    fg.insert(items_df)

# Perform vector similarity search for each query
neighbors = {}
for _, row in queries_df.iterrows():
    query_id = row["query_id"]
    embedding = row["embedding"]
    
    results = fg.find_neighbors(embedding=embedding, col="embedding", k=5)
    
    # Extract item_ids from results
    item_ids = [result[1][0] for result in results]  # result[1][0] is the item_id
    neighbors[query_id] = item_ids

# Write results to submission/answers.json
os.makedirs("submission", exist_ok=True)
with open("submission/answers.json", "w") as f:
    json.dump({"store": "itemsedc1f4", "neighbors": neighbors}, f, indent=2)