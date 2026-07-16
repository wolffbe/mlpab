#!/usr/bin/env python3
"""
Generate recommendations for every user and upload to Hopsworks.
"""

import pandas as pd
import numpy as np
import hopsworks

# Load data
interactions_df = pd.read_csv("data/interactions.csv")
user_embeddings_df = pd.read_csv("data/user_embeddings.csv")
item_embeddings_df = pd.read_csv("data/item_embeddings.csv")

# Parse embeddings
user_embeddings = {}
for _, row in user_embeddings_df.iterrows():
    user_id = row["user_id"]
    embedding = row["e1":"e8"].values.astype(np.float32)
    user_embeddings[user_id] = embedding

item_embeddings = {}
for _, row in item_embeddings_df.iterrows():
    item_id = row["item_id"]
    embedding = row["e1":"e8"].values.astype(np.float32)
    item_embeddings[item_id] = embedding

# Get all unique user_ids and item_ids
user_ids = list(user_embeddings.keys())
item_ids = list(item_embeddings.keys())

# Generate recommendations
recommendations = []
for user_id in user_ids:
    # Get user embedding
    user_embedding = user_embeddings[user_id]
    
    # Get items the user has already interacted with
    interacted_items = interactions_df[interactions_df["user_id"] == user_id]["item_id"].tolist()
    
    # Compute dot products for all items
    scores = []
    for item_id in item_ids:
        if item_id in interacted_items:
            continue
        item_embedding = item_embeddings[item_id]
        score = np.dot(user_embedding, item_embedding)
        scores.append((item_id, score))
    
    # Sort by score (descending) and item_id (ascending for ties)
    scores.sort(key=lambda x: (-x[1], x[0]))
    
    # Take top 5
    top_items = scores[:5]
    
    # Format recommendations
    for rank, (item_id, score) in enumerate(top_items, start=1):
        rec_id = f"{user_id}#{rank}"
        recommendations.append({
            "rec_id": rec_id,
            "user_id": user_id,
            "rank": rank,
            "item_id": item_id
        })

# Convert to DataFrame
recommendations_df = pd.DataFrame(recommendations)

# Upload to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Create feature group
fg = fs.create_feature_group(
    name="recsc757d5",
    version=1,
    description="Top-5 recommendations for every user, ranked by relevance (dot product).",
    primary_key=["rec_id"],
    online_enabled=True,
    expectation_suite=None
)

# Upload data
fg.insert(recommendations_df, write_options={"wait_for_job": True})

print("Recommendations uploaded successfully.")