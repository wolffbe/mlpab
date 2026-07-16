import hopsworks
import pandas as pd
import numpy as np
import os

project = hopsworks.login()
fs = project.get_feature_store()

dataset_api = project.get_dataset_api()

dataset_api.download("Resources/recsys_data/user_embeddings.csv", local_path="/tmp/user_embeddings.csv", overwrite=True)
dataset_api.download("Resources/recsys_data/item_embeddings.csv", local_path="/tmp/item_embeddings.csv", overwrite=True)
dataset_api.download("Resources/recsys_data/interactions.csv", local_path="/tmp/interactions.csv", overwrite=True)

users_df = pd.read_csv("/tmp/user_embeddings.csv")
items_df = pd.read_csv("/tmp/item_embeddings.csv")
interactions_df = pd.read_csv("/tmp/interactions.csv")

emb_cols = [f"e{i}" for i in range(1, 9)]

user_vecs = users_df[emb_cols].values
item_vecs = items_df[emb_cols].values

scores = user_vecs @ item_vecs.T

interacted = set(zip(interactions_df["user_id"], interactions_df["item_id"]))

rows = []
for i, user_id in enumerate(users_df["user_id"]):
    user_scores = []
    for j, item_id in enumerate(items_df["item_id"]):
        if (user_id, item_id) not in interacted:
            user_scores.append((scores[i, j], item_id))
    user_scores.sort(key=lambda x: (-x[0], x[1]))
    for rank, (score, item_id) in enumerate(user_scores[:5], start=1):
        rec_id = f"{user_id}#{rank}"
        rows.append({"rec_id": rec_id, "user_id": user_id, "rank": rank, "item_id": item_id})

recs_df = pd.DataFrame(rows, columns=["rec_id", "user_id", "rank", "item_id"])
print(f"Generated {len(recs_df)} recommendations")
print(recs_df.head(10))

fg = fs.get_or_create_feature_group(
    name="recs8ae03d",
    version=1,
    primary_key=["rec_id"],
    description="Top-5 item recommendations per user based on embedding dot product",
    online_enabled=True,
)

fg.insert(recs_df)
print("Feature group created and data inserted successfully.")
