"""Compute top-5 recommendations per user on the Hopsworks platform.

relevance = dot(user_emb, item_emb); exclude already-interacted items;
rank 1 = highest score; ties broken by ascending item_id; 5 rows per user.
Writes feature group `recsaecaa2` v1 (offline + online).
"""
import hopsworks
import numpy as np
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()

ue = fs.get_feature_group("user_embeddings", version=1).read()
ie = fs.get_feature_group("item_embeddings", version=1).read()
inter = fs.get_feature_group("user_interactions", version=1).read()

ecols = [f"e{i}" for i in range(1, 9)]

ue = ue.sort_values("user_id").reset_index(drop=True)
ie = ie.sort_values("item_id").reset_index(drop=True)

uids = ue["user_id"].tolist()
iids = ie["item_id"].tolist()
U = ue[ecols].to_numpy(dtype=float)   # (n_users, 8)
I = ie[ecols].to_numpy(dtype=float)   # (n_items, 8)
scores = U @ I.T                       # (n_users, n_items)

seen = inter.groupby("user_id")["item_id"].apply(set).to_dict()

rows = []
for ui, u in enumerate(uids):
    excl = seen.get(u, set())
    # (score, item_id) for candidate items not yet interacted with
    cand = [(scores[ui, ii], iids[ii]) for ii in range(len(iids)) if iids[ii] not in excl]
    # highest score first; ties -> ascending item_id
    cand.sort(key=lambda x: (-x[0], x[1]))
    for rank, (_, item_id) in enumerate(cand[:5], start=1):
        rows.append({
            "rec_id": f"{u}#{rank}",
            "user_id": u,
            "rank": int(rank),
            "item_id": item_id,
        })

res = pd.DataFrame(rows, columns=["rec_id", "user_id", "rank", "item_id"])
res["rank"] = res["rank"].astype("int64")
print(f"Built {len(res)} recommendation rows for {len(uids)} users")

target = fs.get_or_create_feature_group(
    name="recsaecaa2",
    version=1,
    description="Top-5 recommended items per user (two-tower dot product, interacted items excluded)",
    primary_key=["rec_id"],
    online_enabled=True,
)
target.insert(res)
print("Inserted into recsaecaa2 v1")
