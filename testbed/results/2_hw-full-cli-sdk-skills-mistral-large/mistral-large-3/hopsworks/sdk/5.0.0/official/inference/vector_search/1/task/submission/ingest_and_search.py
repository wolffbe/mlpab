#!/usr/bin/env python3
import hopsworks
import pandas as pd
import json

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Load data
items_df = pd.read_csv("../data/items.csv")
queries_df = pd.read_csv("../data/queries.csv")

# Parse embeddings
items_df["embedding"] = items_df["embedding"].apply(lambda x: json.loads(x))
queries_df["embedding"] = queries_df["embedding"].apply(lambda x: json.loads(x))

# Create embedding index
from hsfs.embedding import EmbeddingIndex

embedding_index = EmbeddingIndex()
embedding_index.add_embedding(name="embedding", dimension=16)

# Create or get feature group
try:
    fg = fs.get_feature_group("itemsedc1f4", version=1)
    fg.delete()
except:
    pass

fg = fs.create_feature_group(
    name="itemsedc1f4",
    version=1,
    description="Feature group for items with vector embeddings",
    primary_key=["item_id"],
    embedding_index=embedding_index,
    online_enabled=True,
)

# Ingest items
fg.insert(items_df)

# Search for each query
neighbors = {}
for _, row in queries_df.iterrows():
    query_embedding = row["embedding"]
    results = fg.find_neighbors(
        embedding=query_embedding,
        k=5
    )
    neighbors[row["query_id"]] = [result[1][0] for result in results]

# Write results
with open("answers.json", "w") as f:
    json.dump({
        "store": "itemsedc1f4",
        "neighbors": neighbors
    }, f, indent=2)