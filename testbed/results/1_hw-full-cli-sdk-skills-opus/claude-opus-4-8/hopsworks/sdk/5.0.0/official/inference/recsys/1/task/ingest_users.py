import csv, hopsworks

project = hopsworks.login()
fs = project.get_feature_store()

ecols = [f"e{i}" for i in range(1, 9)]

def load(path, floatcols):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            for c in floatcols:
                r[c] = float(r[c])
            rows.append(r)
    return rows

users = load("data/user_embeddings.csv", ecols)
print("user rows", len(users), "sample", users[0])

uemb = fs.get_or_create_feature_group(
    name="user_emb", version=1,
    description="Two-tower user embeddings (8-dim)",
    primary_key=["user_id"], online_enabled=False, statistics_config=False)
uemb.insert(users, wait=True)
print("user_emb id", uemb.id)
print("schema", [(f.name, f.type) for f in uemb.features])
