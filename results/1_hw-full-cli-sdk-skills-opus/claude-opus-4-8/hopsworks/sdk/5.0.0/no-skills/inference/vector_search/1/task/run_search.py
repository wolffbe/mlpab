import json, csv, hopsworks
from hsfs.embedding import EmbeddingIndex
from hsfs.embedding import SimilarityFunctionType
import pandas as pd

STORE = "items5a26f7"

# --- Load data ---
items = []
with open("data/items.csv") as f:
    for r in csv.DictReader(f):
        items.append({"item_id": r["item_id"],
                      "embedding": json.loads(r["embedding"]),
                      "label": r["label"]})
queries = []
with open("data/queries.csv") as f:
    for r in csv.DictReader(f):
        queries.append({"query_id": r["query_id"],
                        "embedding": json.loads(r["embedding"])})
dim = len(items[0]["embedding"])
print("items", len(items), "queries", len(queries), "dim", dim)

proj = hopsworks.login()
fs = proj.get_feature_store()

# --- Embedding index with L2 ---
emb = EmbeddingIndex(index_name=STORE)
emb.add_embedding("embedding", dim, SimilarityFunctionType.L2)

fg = fs.get_or_create_feature_group(
    name=STORE,
    version=1,
    primary_key=["item_id"],
    embedding_index=emb,
    online_enabled=True,
)

df = pd.DataFrame(items)
fg.insert(df, wait=True)
print("inserted")

# --- Vector search ---
neighbors = {}
for q in queries:
    res = fg.find_neighbors(q["embedding"], col="embedding", k=5)
    ids = []
    for score, row in res:
        d = dict(zip([feat.name for feat in fg.features], row))
        ids.append(d["item_id"])
    neighbors[q["query_id"]] = ids
    print(q["query_id"], ids)

out = {"store": STORE, "neighbors": neighbors}
with open("submission/answers.json", "w") as f:
    json.dump(out, f, indent=2)
print("WROTE submission/answers.json")
