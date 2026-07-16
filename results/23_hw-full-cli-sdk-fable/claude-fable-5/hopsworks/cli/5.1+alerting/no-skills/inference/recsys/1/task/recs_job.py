"""Compute top-5 recommendations per user and write them to an online-enabled
feature group `recs48963e` v1. Runs as a Hopsworks PYTHON job."""
import hopsworks
import pandas as pd
import numpy as np

project = hopsworks.login()
ds = project.get_dataset_api()

paths = {}
for f in ["interactions.csv", "user_embeddings.csv", "item_embeddings.csv"]:
    paths[f] = ds.download(f"Resources/recdata48963e/{f}", overwrite=True)

inter = pd.read_csv(paths["interactions.csv"])
users = pd.read_csv(paths["user_embeddings.csv"])
items = pd.read_csv(paths["item_embeddings.csv"])

emb_cols = [f"e{i}" for i in range(1, 9)]
U = users[emb_cols].to_numpy(dtype=float)
V = items[emb_cols].to_numpy(dtype=float)
scores = U @ V.T

item_ids = items["item_id"].tolist()
seen = inter.groupby("user_id")["item_id"].apply(set).to_dict()

rows = []
for ui, uid in enumerate(users["user_id"]):
    excluded = seen.get(uid, set())
    cand = [(item_ids[j], scores[ui, j]) for j in range(len(item_ids))
            if item_ids[j] not in excluded]
    cand.sort(key=lambda t: (-t[1], t[0]))
    for rank, (iid, _) in enumerate(cand[:5], start=1):
        rows.append({"rec_id": f"{uid}#{rank}", "user_id": uid,
                     "rank": rank, "item_id": iid})

df = pd.DataFrame(rows, columns=["rec_id", "user_id", "rank", "item_id"])
df["rank"] = df["rank"].astype("int64")
print(f"Computed {len(df)} rows for {users.shape[0]} users")
print(df.head(10).to_string())

fs = project.get_feature_store()
fg = fs.get_or_create_feature_group(
    name="recs48963e",
    version=1,
    primary_key=["rec_id"],
    online_enabled=True,
    description="Top-5 two-tower dot-product recommendations per user",
)
fg.insert(df, wait=True)
print("Insert complete:", len(df), "rows written to recs48963e v1")
