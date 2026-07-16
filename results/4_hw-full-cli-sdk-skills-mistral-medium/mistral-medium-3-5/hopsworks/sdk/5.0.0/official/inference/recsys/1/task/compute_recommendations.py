#!/usr/bin/env python3
"""
Compute top-5 recommendations for every user using two-tower embeddings.
Excludes already-interacted items, ranks by dot product (descending),
breaks ties by ascending item_id.
"""

import pandas as pd
import numpy as np
import hopsworks

# Read data
interactions = pd.read_csv('data/interactions.csv')
user_embeddings = pd.read_csv('data/user_embeddings.csv')
item_embeddings = pd.read_csv('data/item_embeddings.csv')

# Extract embedding columns (e1..e8)
user_emb_cols = [c for c in user_embeddings.columns if c.startswith('e')]
item_emb_cols = [c for c in item_embeddings.columns if c.startswith('e')]

# Convert to numpy arrays for efficient dot product computation
user_emb_matrix = user_embeddings[user_emb_cols].values  # shape: (n_users, 8)
item_emb_matrix = item_embeddings[item_emb_cols].values  # shape: (n_items, 8)

# Compute all dot products: score[i, j] = dot product of user i and item j
score_matrix = np.dot(user_emb_matrix, item_emb_matrix.T)  # shape: (n_users, n_items)

# Build interaction lookup: for each user, set of items they've interacted with
user_interactions = interactions.groupby('user_id')['item_id'].apply(set).to_dict()

# Map user_id and item_id to indices
idx_to_item_id = {j: iid for j, iid in enumerate(item_embeddings['item_id'])}
idx_to_user_id = {i: uid for i, uid in enumerate(user_embeddings['user_id'])}

# Build recommendations
all_recs = []

for user_idx in range(len(user_embeddings)):
    user_id = idx_to_user_id[user_idx]
    
    # Get items this user has already interacted with
    interacted_items = user_interactions.get(user_id, set())
    
    # Get scores for this user
    scores = score_matrix[user_idx]  # shape: (n_items,)
    
    # Build list of (item_id, score, item_idx) for all items
    candidates = []
    for item_idx in range(len(item_embeddings)):
        item_id = idx_to_item_id[item_idx]
        if item_id not in interacted_items:
            candidates.append((item_id, scores[item_idx], item_idx))
    
    # Sort by score descending, then by item_id ascending for ties
    candidates.sort(key=lambda x: (-x[1], x[0]))
    
    # Take top 5
    top5 = candidates[:5]
    
    # Create rec_id and rank
    for rank, (item_id, score, item_idx) in enumerate(top5, 1):
        rec_id = f"{user_id}#{rank}"
        all_recs.append({
            'rec_id': rec_id,
            'user_id': user_id,
            'rank': rank,
            'item_id': item_id
        })

# Create DataFrame
rec_df = pd.DataFrame(all_recs)

# Verify we have 5 rows per user
print(f"Total recommendations: {len(rec_df)}")
print(f"Expected: {len(user_embeddings) * 5}")
print(f"Users: {len(user_embeddings)}")

# Connect to Hopsworks and create feature table
print("Connecting to Hopsworks...")
hopsworks.login()
fs = hopsworks.get_current_project().get_feature_store()

# Create feature group
fg_name = "recsed2647"
version = 1

# Check if feature group exists, if so delete it first
try:
    fg = fs.get_feature_group(fg_name, version=version)
    print(f"Found existing feature group {fg_name} v{version}, will overwrite")
except Exception as e:
    print(f"Feature group doesn't exist or error: {e}")
    fg = None

# Create the feature group with online enabled
fg = fs.create_feature_group(
    name=fg_name,
    version=version,
    description="Top-5 recommendations for each user based on two-tower embeddings",
    primary_key=['rec_id'],
    online_enabled=True
)

# Insert data
print(f"Inserting {len(rec_df)} rows into feature group...")
fg.insert(rec_df, write_options={"wait_for_job": True})

print(f"Feature table '{fg_name}' v{version} created successfully with {len(rec_df)} rows")
print("\nSample recommendations:")
print(rec_df.head(10))
