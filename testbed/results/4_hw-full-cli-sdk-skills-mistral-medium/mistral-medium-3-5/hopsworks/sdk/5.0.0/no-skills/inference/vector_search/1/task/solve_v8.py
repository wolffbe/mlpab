#!/usr/bin/env python3
"""
Load item embeddings into Hopsworks vector store and perform similarity search.
"""
import json
import pandas as pd
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

# Get or create the feature group with primary key
fg = fs.get_or_create_feature_group(
    name=store_name,
    version=1,
    description="Vector store for items with embeddings",
    online_enabled=True,
    statistics_config=False,
    primary_key=["item_id"]
)
print(f"Feature group: {fg.name}")

# Create embedding index
print("\nCreating embedding index...")
embedding_features = []
for i in range(16):
    emb_feature = hsfs_embedding.EmbeddingFeature(
        name=f"embedding_{i}",
        dimension=16,
        similarity_function_type=hsfs_embedding.SimilarityFunctionType.L2_NORM,
        feature_group=fg
    )
    embedding_features.append(emb_feature)

emb_index = hsfs_embedding.EmbeddingIndex(
    index_name=store_name,
    features=embedding_features,
    col_prefix="embedding"
)

# Update the feature group with the embedding index
fg.embedding_index = emb_index
print("Embedding index created")

# Read items.csv
print("\nReading items.csv...")
items_df = pd.read_csv("data/items.csv")

# Parse the embedding JSON strings into separate columns
print("Parsing embeddings...")
embeddings = items_df['embedding'].apply(json.loads)
for i in range(16):
    items_df[f'embedding_{i}'] = embeddings.apply(lambda x: x[i])

# Drop the original embedding column
items_df = items_df.drop(columns=['embedding'])

print(f"DataFrame shape: {items_df.shape}")

# Insert data into the feature group
print("Inserting data into feature group...")
fg.insert(items_df, write_options={"wait_for_job": True})
print("Data inserted successfully")

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
        # Perform similarity search
        result = fg.find_neighbors(
            embedding=query_embedding,
            k=5
        )
        
        # Extract item_ids from the result
        # Result is a list of tuples: (distance, row_data)
        item_ids = []
        for distance, row in result:
            # row should be a dict with item_id
            if isinstance(row, dict) and 'item_id' in row:
                item_ids.append(row['item_id'])
            else:
                print(f"Warning: unexpected row format: {type(row)}, {row}")
        
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
