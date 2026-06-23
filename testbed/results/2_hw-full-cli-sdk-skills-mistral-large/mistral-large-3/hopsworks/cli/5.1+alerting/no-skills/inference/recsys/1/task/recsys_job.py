#!/usr/bin/env python3
"""
Compute top-5 recommendations for every user:
- relevance = dot product of user embedding and item embedding
- exclude items the user has already interacted with
- rank 1 = highest relevance; break ties by ascending item_id
- write to feature group `recsc757d5`, version 1, with schema:
  rec_id (key), user_id, rank (1..5), item_id
"""

import hopsworks
import pandas as pd
import numpy as np

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Read data
interactions_df = pd.read_csv("/Projects/mlpabb9d930/Resources/recsys_data/interactions.csv")
user_embeddings_df = pd.read_csv("/Projects/mlpabb9d930/Resources/recsys_data/user_embeddings.csv")
item_embeddings_df = pd.read_csv("/Projects/mlpabb9d930/Resources/recsys_data/item_embeddings.csv")

# Melt embeddings for dot product
user_embeddings = user_embeddings_df.set_index("user_id")[["e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8"]].values
item_embeddings = item_embeddings_df.set_index("item_id")[["e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8"]].values

# Compute dot products for all user-item pairs
all_items = item_embeddings_df["item_id"].values
all_users = user_embeddings_df["user_id"].values

# Broadcast dot product
user_embeddings_bc = user_embeddings[:, np.newaxis, :]  # (U, 1, 8)
item_embeddings_bc = item_embeddings[np.newaxis, :, :]  # (1, I, 8)
dot_products = np.sum(user_embeddings_bc * item_embeddings_bc, axis=2)  # (U, I)

# Build DataFrame for ranking
dot_df = pd.DataFrame(dot_products, index=all_users, columns=all_items)

# Exclude interacted items
interacted_items = interactions_df.groupby("user_id")["item_id"].apply(set)
for user_id in all_users:
    interacted = interacted_items.get(user_id, set())
    dot_df.loc[user_id, list(interacted)] = -np.inf

# Rank: top-5 by descending dot product, ascending item_id for ties
def rank_items(row):
    ranked = row.sort_values(ascending=False).reset_index()
    ranked.columns = ["item_id", "score"]
    ranked["rank"] = ranked["score"].rank(method="first", ascending=False).astype(int)
    ranked = ranked.sort_values(["rank", "item_id"]).head(5)
    ranked["user_id"] = row.name
    return ranked

ranked_dfs = []
for user_id in all_users:
    ranked_dfs.append(rank_items(dot_df.loc[user_id]))

result_df = pd.concat(ranked_dfs)
result_df["rec_id"] = result_df["user_id"] + "#" + result_df["rank"].astype(str)
result_df = result_df[["rec_id", "user_id", "rank", "item_id"]]

# Write to feature group
fg = fs.get_or_create_feature_group(
    name="recsc757d5",
    version=1,
    primary_key=["rec_id"],
    description="Top-5 recommendations for every user, ranked by dot product of embeddings (excludes interacted items).",
    online_enabled=True,
)
fg.insert(result_df)