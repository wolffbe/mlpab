"""Runs ON the Hopsworks platform as a PYTHON job.

Computes top-5 two-tower recommendations per user (dot-product relevance,
excluding prior interactions, ties broken by ascending item_id) and writes the
result to the online-enabled feature group `recsfd473b` v1.
"""
import hopsworks
import pandas as pd
import numpy as np
from hsfs.feature import Feature

project = hopsworks.login()
fs = project.get_feature_store()
ds = project.get_dataset_api()

base = "Resources/recs_data"
for fn in ["user_embeddings.csv", "item_embeddings.csv", "interactions.csv"]:
    ds.download(f"{base}/{fn}", fn, overwrite=True)

ecols = [f"e{i}" for i in range(1, 9)]
users = pd.read_csv("user_embeddings.csv", dtype={"user_id": str})
items = pd.read_csv("item_embeddings.csv", dtype={"item_id": str})
inter = pd.read_csv("interactions.csv", dtype={"user_id": str, "item_id": str})
print("loaded", users.shape, items.shape, inter.shape)

U = users[ecols].to_numpy(dtype=float)        # (n_users, 8)
I = items[ecols].to_numpy(dtype=float)        # (n_items, 8)
scores = U.dot(I.T)                            # (n_users, n_items) dot products

seen = {}
for uid, iid in zip(inter.user_id, inter.item_id):
    seen.setdefault(uid, set()).add(iid)

uids = users.user_id.tolist()
iids = items.item_id.tolist()

recs = []
for ui, uid in enumerate(uids):
    s = seen.get(uid, set())
    cand = [(iids[ii], float(scores[ui, ii])) for ii in range(len(iids)) if iids[ii] not in s]
    # relevance desc, then item_id ascending for exact ties
    cand.sort(key=lambda x: (-x[1], x[0]))
    for rank, (iid, sc) in enumerate(cand[:5], start=1):
        recs.append({"rec_id": f"{uid}#{rank}", "user_id": uid, "rank": int(rank), "item_id": iid})

recs_df = pd.DataFrame(recs, columns=["rec_id", "user_id", "rank", "item_id"])
recs_df["rank"] = recs_df["rank"].astype("int32")  # maps to hsfs 'int'
print("computed recs:", recs_df.shape)
print(recs_df.head(7).to_string())

fg = fs.get_or_create_feature_group(
    name="recsfd473b", version=1,
    description="Top-5 two-tower recommendations per user (dot-product relevance, prior interactions excluded)",
    primary_key=["rec_id"],
    features=[
        Feature("rec_id", "string", description="Record key formatted <user_id>#<rank>"),
        Feature("user_id", "string", description="User id"),
        Feature("rank", "int", description="Rank 1..5 (1 = highest relevance)"),
        Feature("item_id", "string", description="Recommended item id"),
    ],
    online_enabled=True, stream=True, statistics_config=False)

fg.insert(recs_df, wait=True)
print("INSERTED", len(recs_df), "rows into recsfd473b; fg id", fg.id)
