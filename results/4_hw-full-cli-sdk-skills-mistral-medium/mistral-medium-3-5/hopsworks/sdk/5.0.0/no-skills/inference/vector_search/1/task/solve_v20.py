#!/usr/bin/env python3
"""
Load item embeddings into Hopsworks vector store and perform similarity search.
"""
import json
import time
import pandas as pd
import numpy as np
import hopsworks
from hopsworks.hsfs import embedding as hsfs_embedding

# Connect to Hopsworks
hopsworks.login()

# Get the project
project = hopsworks.project.Project()

# Create or get the feature store
fs = project.get_feature_store()

# Name for our vector store
store_name = "itemsb84082"

# Delete existing feature group to start fresh
try:
    fg = fs.get_feature_group(store_name, version=1)
    fg.delete()
    print(f"Deleted existing feature group {store_name}")
except:
    pass

# Create embedding feature with L2 similarity
emb_feature = hsfs_embedding.EmbeddingFeature(
    name="embedding",
    dimension=16,
    similarity_function_type=hsfs_embedding.SimilarityFunctionType.L2
)

# Create embedding index
emb_index = hsfs_embedding.EmbeddingIndex(
    index_name=store_name,
    features=[emb_feature],
    col_prefix=""
)

# Get or create the feature group with primary key and embedding index
fg = fs.get_or_create_feature_group(
    name=store_name,
    version=1,
    description="Vector store for items with embeddings",
    online_enabled=True,
    statistics_config=False,
    primary_key=["item_id"],
    embedding_index=emb_index
)
print(f"Feature group: {fg.name}")

# Read items.csv
print("\nReading items.csv...")
items_df = pd.read_csv("data/items.csv")

# Parse the embedding JSON strings into numpy float32 arrays
print("Parsing embeddings...")
items_df['embedding'] = items_df['embedding'].apply(lambda x: np.array(json.loads(x), dtype=np.float32))

print(f"DataFrame shape: {items_df.shape}")

# Insert data into the feature group
print("Inserting data into feature group...")
fg.insert(items_df, write_options={"wait_for_job": True})
print("Data inserted successfully")

# Wait for the index to be built
print("\nWaiting for vector index to be built...")
time.sleep(30)

# Now read queries.csv
print("\nReading queries.csv...")
queries_df = pd.read_csv("data/queries.csv")
queries_df['embedding'] = queries_df['embedding'].apply(json.loads)

print(f"Loaded {len(queries_df)} queries")

# Perform similarity search for each query using find_neighbors
print("\nPerforming similarity search...")
neighbors = {}

for _, query_row in queries_df.iterrows():
    query_id = query_row["query_id"]
    query_embedding = query_row["embedding"]
    
    print(f"Processing query {query_id}...")
    
    try:
        # Perform similarity search with options to specify distance metric
        result = fg.find_neighbors(
            embedding=list(query_embedding),
            k=5,
            options={"distance_metric": "l2"}  # Try to specify L2 distance explicitly
        )
        
        # Extract item_ids from the result
        # Result is a list of tuples: (distance, row_data)
        # where row_data is a list: [item_id, embedding, label]
        item_ids = []
        for distance, row_data in result:
            if isinstance(row_data, list) and len(row_data) > 0:
                item_ids.append(row_data[0])
            elif isinstance(row_data, dict) and 'item_id' in row_data:
                item_ids.append(row_data['item_id'])
            else:
                print(f"Warning: unexpected row_data format: {type(row_data)}")
        
        neighbors[query_id] = item_ids
        print(f"  Found {len(item_ids)} neighbors: {item_ids}")
        
    except Exception as e:
        print(f"Error processing query {query_id}: {e}")
        import traceback
        traceback.print_exc()
        neighbors[query_id] = []

# Write the results
output = {
    "store": store_name,
    "neighbors": neighbors
}

print("\nWriting results to submission/answers.json...")
with open("submission/answers.json", "w") as f:
    json.dump(output, f, indent=2)

print("Done!")
