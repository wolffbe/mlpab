import csv
import inspect
import json
import os

os.environ.pop("NO_PROXY", None)
os.environ.pop("no_proxy", None)

import hopsworks
import pandas as pd
import hsfs.embedding as embedding

project = hopsworks.login()
fs = project.get_feature_store()
print("project:", project.name)

items = []
with open("data/items.csv") as f:
    for row in csv.DictReader(f):
        items.append(
            {
                "item_id": row["item_id"],
                "embedding": json.loads(row["embedding"]),
                "label": row["label"],
            }
        )
print("items:", len(items))

emb_index = embedding.EmbeddingIndex(index_name="itemsf57ff6_index")
emb_index.add_embedding(
    "embedding", 16, similarity_function_type=embedding.SimilarityFunctionType.L2
)

fg = fs.get_or_create_feature_group(
    name="itemsf57ff6",
    version=1,
    primary_key=["item_id"],
    online_enabled=True,
    embedding_index=emb_index,
    description="items with 16-dim embeddings for vector search",
)
print("find_neighbors sig:", inspect.signature(fg.find_neighbors))

df = pd.DataFrame(items)
fg.insert(df, wait=True)
print("inserted")
