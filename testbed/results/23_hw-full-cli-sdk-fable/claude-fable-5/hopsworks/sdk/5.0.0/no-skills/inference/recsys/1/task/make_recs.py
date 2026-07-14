import csv
import os

# The sandbox only allows outbound traffic via the localhost proxy; the
# platform host sits in 10.0.0.0/8 which NO_PROXY would bypass.
for var in ("NO_PROXY", "no_proxy"):
    os.environ.pop(var, None)

import hopsworks
import pandas as pd

# Load inputs
def load_emb(path, key):
    d = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            d[row[key]] = [float(row[f"e{i}"]) for i in range(1, 9)]
    return d

users = load_emb("data/user_embeddings.csv", "user_id")
items = load_emb("data/item_embeddings.csv", "item_id")

seen = {}
with open("data/interactions.csv") as f:
    for row in csv.DictReader(f):
        seen.setdefault(row["user_id"], set()).add(row["item_id"])

rows = []
for user_id in sorted(users):
    uvec = users[user_id]
    scored = []
    for item_id, ivec in items.items():
        if item_id in seen.get(user_id, set()):
            continue
        score = sum(a * b for a, b in zip(uvec, ivec))
        scored.append((-score, item_id))
    scored.sort()  # highest score first, ties by ascending item_id
    for rank, (_, item_id) in enumerate(scored[:5], start=1):
        rows.append({
            "rec_id": f"{user_id}#{rank}",
            "user_id": user_id,
            "rank": rank,
            "item_id": item_id,
        })

df = pd.DataFrame(rows, columns=["rec_id", "user_id", "rank", "item_id"])
df["rank"] = df["rank"].astype("int64")
print(df.head(10))
print("total rows:", len(df))

project = hopsworks.login()
fs = project.get_feature_store()

fg = fs.get_or_create_feature_group(
    name="recs48963e",
    version=1,
    primary_key=["rec_id"],
    online_enabled=True,
    description="Top-5 recommended items per user (two-tower dot product)",
)
fg.insert(df, wait=True)
print("insert complete")
