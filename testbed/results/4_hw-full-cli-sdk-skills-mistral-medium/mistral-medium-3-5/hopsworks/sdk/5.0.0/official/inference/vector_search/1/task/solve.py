#!/usr/bin/env python3
import json
import csv
import os
import hopsworks
import pandas as pd
from hsfs.embedding import EmbeddingIndex, SimilarityFunctionType

# Read items
items = []
with open('data/items.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        items.append({
            'item_id': row['item_id'],
            'embedding': json.loads(row['embedding']),
            'label': row['label']
        })

print(f"Loaded {len(items)} items")

# Read queries
queries = []
with open('data/queries.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        queries.append({
            'query_id': row['query_id'],
            'embedding': json.loads(row['embedding'])
        })

print(f"Loaded {len(queries)} queries")

# Connect to Hopsworks
print("Connecting to Hopsworks...")
hopsworks.login()

# Get the project and feature store
project = hopsworks.get_current_project()
fs = project.get_feature_store()

store_name = "itemsb84082"

# Create embedding index
print("Creating embedding index...")
embedding_index = EmbeddingIndex(
    index_name=store_name,
    col_prefix="embedding"
)

# Add embedding feature
embedding_index.add_embedding(
    name="embedding",
    dimension=16,
    similarity_function_type=SimilarityFunctionType.L2
)

# Create feature group with embedding index
print(f"Creating feature group '{store_name}'...")
fg = fs.get_or_create_feature_group(
    name=store_name,
    version=1,
    description="Vector embeddings for items",
    embedding_index=embedding_index,
    online_enabled=True,
    primary_key=['item_id']
)

# Prepare data for insertion
# Each row needs to have the embedding as a list
feature_data = []
for item in items:
    feature_data.append({
        'item_id': item['item_id'],
        'label': item['label'],
        'embedding': item['embedding']
    })

# Convert to pandas DataFrame
feature_df = pd.DataFrame(feature_data)

# Insert data into the feature group
print("Inserting items into feature store...")
fg.insert(feature_df, write_options={"wait_for_job": True})

print("Data inserted successfully")

# Now perform vector similarity search for each query
neighbors = {}

for query in queries:
    query_id = query['query_id']
    query_embedding = query['embedding']
    
    print(f"Processing query {query_id}...")
    
    try:
        # Use find_neighbors to get the top-5 nearest items
        results = fg.find_neighbors(
            embedding=query_embedding,
            k=5
        )
        
        # Extract item IDs from results
        # Results are tuples of (similarity_score, [feature_values])
        # We need to get the item_id from each result
        item_ids = []
        for score, features in results:
            # features is a list of feature values for that row
            # The order should be: item_id, label, embedding
            item_id = features[0]  # First column is item_id
            item_ids.append(item_id)
        
        neighbors[query_id] = item_ids
        print(f"Query {query_id}: {item_ids}")
        
    except Exception as e:
        print(f"Error processing query {query_id}: {e}")
        import traceback
        traceback.print_exc()
        # Try to continue with next query
        continue

# Write results
result = {
    "store": store_name,
    "neighbors": neighbors
}

os.makedirs('submission', exist_ok=True)
with open('submission/answers.json', 'w') as f:
    json.dump(result, f, indent=2)

print(f"\nResults written to submission/answers.json")
print(f"Store: {store_name}")
print(f"Queries processed: {len(neighbors)}")
