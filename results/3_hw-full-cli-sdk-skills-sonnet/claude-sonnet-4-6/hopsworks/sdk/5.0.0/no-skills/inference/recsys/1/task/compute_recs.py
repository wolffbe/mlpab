import csv
import hopsworks

# Read user embeddings
user_embs = {}
with open("data/user_embeddings.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        uid = row["user_id"]
        emb = [float(row[f"e{i}"]) for i in range(1, 9)]
        user_embs[uid] = emb

# Read item embeddings
item_embs = {}
with open("data/item_embeddings.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        iid = row["item_id"]
        emb = [float(row[f"e{i}"]) for i in range(1, 9)]
        item_embs[iid] = emb

# Read interactions (already seen items per user)
interacted = {}
with open("data/interactions.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        uid = row["user_id"]
        iid = row["item_id"]
        if uid not in interacted:
            interacted[uid] = set()
        interacted[uid].add(iid)

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

# Compute recommendations
all_recs = []
all_items = sorted(item_embs.keys())

for uid in sorted(user_embs.keys()):
    u_emb = user_embs[uid]
    seen = interacted.get(uid, set())

    # Score all unseen items
    scores = []
    for iid in all_items:
        if iid not in seen:
            score = dot(u_emb, item_embs[iid])
            scores.append((score, iid))

    # Sort: descending by score, then ascending by item_id on tie
    scores.sort(key=lambda x: (-x[0], x[1]))

    # Take top 5
    top5 = scores[:5]
    for rank_idx, (score, iid) in enumerate(top5, start=1):
        rec_id = f"{uid}#{rank_idx}"
        all_recs.append({
            "rec_id": rec_id,
            "user_id": uid,
            "rank": rank_idx,
            "item_id": iid,
        })

print(f"Total recommendations: {len(all_recs)}")
print("Sample:", all_recs[:3])

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Create feature group
import pandas as pd
df = pd.DataFrame(all_recs)
df["rank"] = df["rank"].astype(int)
print(df.head(10))
print(df.dtypes)

fg = fs.get_or_create_feature_group(
    name="recs8ae03d",
    version=1,
    primary_key=["rec_id"],
    online_enabled=True,
    description="Top-5 recommendations per user based on two-tower embeddings",
)

fg.insert(df)
print("Done inserting data into feature group recs8ae03d v1")
