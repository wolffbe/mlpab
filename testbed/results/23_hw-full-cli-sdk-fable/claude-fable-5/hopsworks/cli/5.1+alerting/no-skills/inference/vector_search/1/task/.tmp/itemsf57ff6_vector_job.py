"""Platform job: build embedding FG itemsf57ff6 and run L2 KNN for all queries."""
import json
import os

import pandas as pd
import hopsworks
from hsfs import embedding

project = hopsworks.login()
fs = project.get_feature_store()
ds_api = project.get_dataset_api()

items_local = ds_api.download("Resources/itemsf57ff6/items.csv", overwrite=True)
queries_local = ds_api.download("Resources/itemsf57ff6/queries.csv", overwrite=True)

items = pd.read_csv(items_local)
items["embedding"] = items["embedding"].apply(json.loads)

index = embedding.EmbeddingIndex()
index.add_embedding(
    "embedding", 16, similarity_function_type=embedding.SimilarityFunctionType.L2
)

fg = fs.get_or_create_feature_group(
    name="itemsf57ff6",
    version=1,
    primary_key=["item_id"],
    online_enabled=True,
    embedding_index=index,
    description="Item embeddings for L2 vector search",
)
fg.insert(items, wait=True)
print("Inserted rows:", len(items))

feature_names = [f.name for f in fg.features]
item_id_idx = feature_names.index("item_id")

queries = pd.read_csv(queries_local)
neighbors = {}
for _, row in queries.iterrows():
    vec = json.loads(row["embedding"])
    results = fg.find_neighbors(vec, col="embedding", k=20)
    ids = [r[1][item_id_idx] for r in results][:5]
    neighbors[row["query_id"]] = ids
    print(row["query_id"], ids)

out = {"store": "itemsf57ff6", "neighbors": neighbors}
with open("answers.json", "w") as f:
    json.dump(out, f, indent=1)
ds_api.upload("answers.json", "Resources/itemsf57ff6", overwrite=True)
print("DONE", json.dumps(out))
