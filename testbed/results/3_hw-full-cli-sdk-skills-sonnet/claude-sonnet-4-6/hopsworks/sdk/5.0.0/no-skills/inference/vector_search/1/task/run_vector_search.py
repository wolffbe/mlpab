import hopsworks
import hsfs
from hsfs.embedding import EmbeddingIndex, SimilarityFunctionType
import pandas as pd
import json
import os

# Read data
items_df = pd.read_csv('data/items.csv')
queries_df = pd.read_csv('data/queries.csv')

print(f"Items: {len(items_df)}, Queries: {len(queries_df)}")

# Parse embeddings
items_df['embedding'] = items_df['embedding'].apply(json.loads)
queries_df['embedding'] = queries_df['embedding'].apply(json.loads)

print("Sample item embedding length:", len(items_df['embedding'].iloc[0]))

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()
print("Connected to feature store:", fs.name)

# Create or get feature group with embedding index
FG_NAME = 'itemsaa6793'
INDEX_NAME = 'itemsaa6793'
EMBEDDING_DIM = 16

# Try to get existing feature group first
fg = fs.get_feature_group(FG_NAME, version=1)

if fg is None:
    print("Creating feature group with embedding index...")

    embedding_index = EmbeddingIndex(index_name=INDEX_NAME)
    embedding_index.add_embedding(
        name='embedding',
        dimension=EMBEDDING_DIM,
        similarity_function_type=SimilarityFunctionType.L2
    )

    fg = fs.create_feature_group(
        name=FG_NAME,
        version=1,
        description='Items with vector embeddings for similarity search',
        primary_key=['item_id'],
        online_enabled=True,
        embedding_index=embedding_index
    )

    # Insert data
    print("Inserting items into feature group...")
    insert_df = items_df[['item_id', 'embedding', 'label']].copy()
    fg.insert(insert_df)
    print(f"Inserted {len(insert_df)} items")

    # Wait for indexing
    import time
    print("Waiting for data to be indexed...")
    time.sleep(30)
else:
    print(f"Feature group '{FG_NAME}' already exists, reusing it.")
    print(f"FG type: {type(fg)}")
    print(f"FG features: {fg.features}")

# Now do vector similarity search for each query
print("Running vector similarity searches...")

neighbors = {}

for _, row in queries_df.iterrows():
    query_id = row['query_id']
    query_embedding = row['embedding']

    # Find top-5 nearest neighbors using ANN search
    # Returns list of (score, [feature_values...])
    results = fg.find_neighbors(
        embedding=query_embedding,
        k=5,
        col='embedding'
    )

    print(f"  {query_id} raw results: {results[:2] if results else []}")
    # Extract item_ids from results - need to figure out order of features
    item_ids = []
    for score, feature_values in results:
        # feature_values is a list of feature values in feature order
        # We need item_id which is the primary key
        item_ids.append(str(feature_values[0]) if feature_values else None)
    neighbors[query_id] = item_ids
    print(f"  {query_id}: {item_ids}")

# Write output
os.makedirs('submission', exist_ok=True)
answer = {
    'store': FG_NAME,
    'neighbors': neighbors
}

with open('submission/answers.json', 'w') as f:
    json.dump(answer, f, indent=2)

print(f"\nWrote submission/answers.json")
print(f"Total queries processed: {len(neighbors)}")
