import csv
import hopsworks

# Load data
def load_embeddings(path, key_col):
    data = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row[key_col]
            emb = [float(row[f'e{i}']) for i in range(1, 9)]
            data[key] = emb
    return data

def load_interactions(path):
    interacted = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row['user_id']
            iid = row['item_id']
            if uid not in interacted:
                interacted[uid] = set()
            interacted[uid].add(iid)
    return interacted

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

user_embs = load_embeddings('data/user_embeddings.csv', 'user_id')
item_embs = load_embeddings('data/item_embeddings.csv', 'item_id')
interacted = load_interactions('data/interactions.csv')

item_ids = sorted(item_embs.keys())

records = []
for uid in sorted(user_embs.keys()):
    u_emb = user_embs[uid]
    already = interacted.get(uid, set())
    scores = []
    for iid in item_ids:
        if iid not in already:
            s = dot(u_emb, item_embs[iid])
            scores.append((s, iid))
    # Sort: descending score, ascending item_id on tie
    scores.sort(key=lambda x: (-x[0], x[1]))
    top5 = scores[:5]
    for rank, (score, iid) in enumerate(top5, start=1):
        rec_id = f"{uid}#{rank}"
        records.append({'rec_id': rec_id, 'user_id': uid, 'rank': rank, 'item_id': iid})

print(f"Total records: {len(records)}")
print("Sample:", records[:3])

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

import pandas as pd
df = pd.DataFrame(records)
df['rank'] = df['rank'].astype(int)
print(df.dtypes)
print(df.head())

# Create feature group
fg = fs.get_or_create_feature_group(
    name="recs8ae03d",
    version=1,
    primary_key=["rec_id"],
    online_enabled=True,
    description="Top-5 recommendations per user based on two-tower dot product"
)

fg.insert(df)
print("Done! Feature group recs8ae03d created and populated.")
