#!/usr/bin/env python3
"""
Load item embeddings into Hopsworks vector store and perform similarity search.
"""
import os
import json
import csv
import hopsworks

# Connect to Hopsworks
hopsworks.login()

# Get the project
project = hopsworks.project.Project()

# Create or get the feature store
fs = project.get_feature_store()

# Name for our vector store
store_name = "itemsb84082"

# Check if the feature group already exists
fg = None
try:
    fg = fs.get_feature_group(store_name, version=1)
    print(f"Feature group {store_name} already exists")
except Exception as e:
    print(f"Feature group doesn't exist, creating: {e}")
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

if fg is None:
    print("ERROR: Could not create or get feature group")
    exit(1)

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

# Perform similarity search for each query using find_neighbors
print("\nPerforming similarity search...")
neighbors = {}

for query in queries:
    query_id = query["query_id"]
    query_embedding = query["embedding"]
    
    print(f"Processing query {query_id}...")
    
    try:
        # Perform similarity search
        # The find_neighbors method expects a list of floats
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
