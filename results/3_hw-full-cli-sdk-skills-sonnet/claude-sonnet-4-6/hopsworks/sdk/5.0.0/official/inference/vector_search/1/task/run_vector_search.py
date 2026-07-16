import hopsworks
import pandas as pd
import json
import os
import time

# Connect to Hopsworks
print("Connecting to Hopsworks...")
project = hopsworks.login()
fs = project.get_feature_store()

# Load data
print("Loading data...")
items_df = pd.read_csv("data/items.csv")
queries_df = pd.read_csv("data/queries.csv")

items_df['embedding'] = items_df['embedding'].apply(json.loads)
queries_df['embedding'] = queries_df['embedding'].apply(json.loads)

print(f"Loaded {len(items_df)} items and {len(queries_df)} queries")
print(f"Embedding dimension: {len(items_df['embedding'].iloc[0])}")

from hsfs.embedding import EmbeddingIndex, EmbeddingFeature, SimilarityFunctionType

STORE_NAME = "itemsaa6793"
DIM = 16

# Get or create the feature group
fg = fs.get_feature_group(STORE_NAME, version=1)

if fg is None:
    print(f"Creating feature group '{STORE_NAME}'...")
    embedding_index = EmbeddingIndex(index_name=STORE_NAME)
    embedding_index.add_embedding(
        name="embedding",
        dimension=DIM,
        similarity_function_type=SimilarityFunctionType.L2,
    )

    fg = fs.create_feature_group(
        name=STORE_NAME,
        version=1,
        description="Items with vector embeddings for similarity search",
        primary_key=["item_id"],
        online_enabled=True,
        embedding_index=embedding_index,
    )
    print("Inserting items...")
    fg.insert(items_df, write_options={"wait_for_job": True})
    print("Items inserted. Waiting for index to be ready...")
    time.sleep(15)
else:
    print(f"Feature group '{STORE_NAME}' already exists, using it.")

# Run vector search for all queries
print("Running vector search for all queries...")
neighbors = {}

for _, row in queries_df.iterrows():
    query_id = row['query_id']
    embedding = row['embedding']

    results = fg.find_neighbors(embedding, k=5)
    # results is list of (score, [feature_values]) where features are in column order
    # We need item_id which is the first column (primary key)

    # Get column names to find item_id position
    col_names = [f.name for f in fg.columns]
    item_id_idx = col_names.index('item_id')

    item_ids = [r[1][item_id_idx] for r in results]
    neighbors[query_id] = item_ids
    print(f"  {query_id}: {item_ids}")

# Write submission
os.makedirs("submission", exist_ok=True)
answer = {
    "store": STORE_NAME,
    "neighbors": neighbors
}

with open("submission/answers.json", "w") as f:
    json.dump(answer, f, indent=2)

print("\nDone! Written submission/answers.json")
print(f"Store name: {STORE_NAME}")
print(f"Queries answered: {len(neighbors)}")
