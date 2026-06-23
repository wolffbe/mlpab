"""Runs ON the Hopsworks platform as a PYTHON job.

Reads the ingested embedding/interaction feature groups, computes the top-5
dot-product recommendations per user (excluding already-interacted items,
ties broken by ascending item_id), and writes them to the online-enabled
`recsfd473b` feature group.
"""
import hopsworks

proj = hopsworks.login()
fs = proj.get_feature_store()

ecols = [f"e{k}" for k in range(1, 9)]

users = fs.get_feature_group("recsfd473b_user_emb", version=1).read()
items = fs.get_feature_group("recsfd473b_item_emb", version=1).read()
inter = fs.get_feature_group("recsfd473b_interactions", version=1).read()
print("read shapes:", users.shape, items.shape, inter.shape, flush=True)

interacted = inter.groupby("user_id")["item_id"].apply(set).to_dict()

item_ids = items["item_id"].tolist()
item_mat = items[ecols].astype(float).values  # on-platform numpy compute

rows = []
for _, urow in users.iterrows():
    uid = urow["user_id"]
    uvec = urow[ecols].astype(float).values
    seen = interacted.get(uid, set())
    scored = []
    for idx, iid in enumerate(item_ids):
        if iid in seen:
            continue
        score = float((item_mat[idx] * uvec).sum())
        scored.append((score, iid))
    # rank 1 = highest score; exact ties broken by ascending item_id
    scored.sort(key=lambda t: (-t[0], t[1]))
    for rank, (score, iid) in enumerate(scored[:5], start=1):
        rows.append({
            "rec_id": f"{uid}#{rank}",
            "user_id": uid,
            "rank": int(rank),
            "item_id": iid,
        })

import pandas as pd
recs = pd.DataFrame(rows, columns=["rec_id", "user_id", "rank", "item_id"])
recs["rank"] = recs["rank"].astype("int64")
print("recs rows:", len(recs), "users:", recs["user_id"].nunique(), flush=True)
print(recs.head(12).to_string(), flush=True)

recs_fg = fs.get_or_create_feature_group(
    name="recsfd473b",
    version=1,
    primary_key=["rec_id"],
    online_enabled=True,
    description="Top-5 dot-product recommendations per user (excludes interacted items).",
)
recs_fg.insert(recs, write_options={"wait_for_job": True})
print("INSERT DONE rows:", len(recs), flush=True)
