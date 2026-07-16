import hopsworks
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()

users = fs.get_feature_group("user_emb", version=1).read()
items = fs.get_feature_group("item_emb", version=1).read()
inter = fs.get_feature_group("interactions", version=1).read()

ecols = [f"e{i}" for i in range(1, 9)]

# Cross join every user with every item.
users = users.copy()
items = items.copy()
users["_k"] = 1
items["_k"] = 1
cross = users.merge(items, on="_k", suffixes=("_u", "_i"))

# relevance = dot product of user and item embedding
score = sum(cross[f"{c}_u"] * cross[f"{c}_i"] for c in ecols)
cross["score"] = score
cross = cross[["user_id", "item_id", "score"]]

# Exclude items the user has already interacted with.
seen = set(zip(inter["user_id"].tolist(), inter["item_id"].tolist()))
keep = [(u, i) not in seen for u, i in zip(cross["user_id"].tolist(), cross["item_id"].tolist())]
cross = cross[keep]

# Rank: highest score first, break exact ties by ascending item_id.
cross = cross.sort_values(
    ["user_id", "score", "item_id"], ascending=[True, False, True]
).reset_index(drop=True)
cross["rank"] = cross.groupby("user_id").cumcount() + 1

top = cross[cross["rank"] <= 5].copy()
top["rank"] = top["rank"].astype(int)
top["rec_id"] = top["user_id"].astype(str) + "#" + top["rank"].astype(str)
out = top[["rec_id", "user_id", "rank", "item_id"]].reset_index(drop=True)

print("ROWS", len(out), "USERS", out["user_id"].nunique())
print(out.head(10).to_string())

recs = fs.get_or_create_feature_group(
    name="recs20d906",
    version=1,
    description="Top-5 dot-product recommendations per user (interacted items excluded)",
    primary_key=["rec_id"],
    online_enabled=True,
)
recs.insert(out, write_options={"wait_for_job": True})
print("DONE inserted", len(out), "rows into recs20d906 v1")
