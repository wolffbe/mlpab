#!/usr/bin/env python3
"""
Load item embeddings into Hopsworks vector store and perform similarity search.
"""
import os
import json
import csv
import numpy as np
import hopsworks

# Connect to Hopsworks
hopsworks.login()

# Get the project
project = hopsworks.project()

# Create or get the feature store
fs = project.get_feature_store()

# Name for our vector store
store_name = "itemsb84082"

# Check if the feature group already exists, if not create it
try:
    fg = fs.get_feature_group(store_name, version=1)
    print(f"Feature group {store_name} already exists")
except:
    # Create the feature group
    fg = fs.create_feature_group(
        name=store_name,
        version=1,
        description="Vector store for items with embeddings",
        online_enabled=True,
        statistics_config=False,
        event_time_computation_mode="none"
    )
    print(f"Created feature group {store_name}")

# Read items.csv and load into the feature group
print("Reading items.csv...")
items = []
with open("data/items.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        items.append({
            "item_id": row["item_id"],
            "embedding": json.loads(row["embedding"]),
            "label": row["label"]
        })

print(f"Loaded {len(items)} items")

# Prepare data for insertion
# We need to flatten the embedding array into individual columns
# Hopsworks feature groups work with tabular data
# Let's create a list of dicts with item_id, label, and embedding_0, embedding_1, ..., embedding_15
print("Preparing data for insertion...")
feature_data = []
for item in items:
    row = {"item_id": [item["item_id"]], "label": [item["label"]]}
    for i, val in enumerate(item["embedding"]):
        row[f"embedding_{i}"] = [val]
    feature_data.append(row)

# Insert data into the feature group
print("Inserting data into feature group...")
fg.insert(feature_data, write_options={"wait_for_job": True})
print("Data inserted successfully")

# Now read queries.csv
print("\nReading queries.csv...")
queries = []
with open("data/queries.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        queries.append({
            "query_id": row["query_id"],
            "embedding": json.loads(row["embedding"])
        })

print(f"Loaded {len(queries)} queries")

# Now we need to perform vector similarity search
# Let's check what vector search capabilities are available
print("\nChecking Hopsworks vector search capabilities...")

# Try to use the vector search API
# In Hopsworks, we can use the feature store's similarity search
# First, let's check if there's a vector index

# Get the feature view or use the feature group directly
# We need to create a feature view for vector search
try:
    fv = fs.get_feature_view(store_name, version=1)
    print(f"Feature view {store_name} already exists")
except:
    # Create a feature view
    query = fg.select_all()
    fv = fs.create_feature_view(
        name=store_name,
        version=1,
        description="Feature view for vector search",
        labels=["label"],
        query=query
    )
    print(f"Created feature view {store_name}")

# Now try to perform similarity search
# Hopsworks has a similarity_search method on feature views
print("\nPerforming similarity search...")

neighbors = {}

# For each query, perform similarity search
for query in queries:
    query_id = query["query_id"]
    query_embedding = query["embedding"]
    
    print(f"Processing query {query_id}...")
    
    # Convert query embedding to the format expected by Hopsworks
    # We need to create a DataFrame-like structure
    query_df = {
        f"embedding_{i}": [query_embedding[i]] for i in range(len(query_embedding))
    }
    
    try:
        # Perform similarity search
        # The similarity_search method expects a query and returns nearest neighbors
        # We need to check the exact API
        result = fv.similarity_search(
            query=query_df,
            top_k=5,
            distance_metric="euclidean"
        )
        
        # Extract item_ids from the result
        # The result should contain the nearest neighbors
        # Let's inspect the result structure
        print(f"Result type: {type(result)}")
        print(f"Result: {result}")
        
        # Assuming result is a DataFrame with item_id column
        if hasattr(result, 'to_dict'):
            result_dict = result.to_dict()
            if 'item_id' in result_dict:
                item_ids = result_dict['item_id']
                neighbors[query_id] = item_ids
            else:
                print(f"Warning: 'item_id' not in result columns: {result_dict.keys()}")
                # Try to find the ID column
                for key in result_dict.keys():
                    if 'id' in key.lower():
                        neighbors[query_id] = result_dict[key]
                        break
        elif isinstance(result, list):
            # If it's a list of results
            item_ids = []
            for r in result:
                if isinstance(r, dict) and 'item_id' in r:
                    item_ids.append(r['item_id'])
                elif hasattr(r, 'item_id'):
                    item_ids.append(r.item_id)
            neighbors[query_id] = item_ids
        else:
            print(f"Unexpected result type: {type(result)}")
            neighbors[query_id] = []
            
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
