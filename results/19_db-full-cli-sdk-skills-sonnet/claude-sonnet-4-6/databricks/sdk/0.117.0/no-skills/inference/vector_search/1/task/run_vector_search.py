import os
import csv
import json
import time
from datetime import timedelta

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import vectorsearch

# Environment config
SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]  # workspace.mlpab603fc4
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]   # mlpab603fc4

STORE_NAME = "items9110bd"
ENDPOINT_NAME = f"{PREFIX}_{STORE_NAME}"
INDEX_NAME = f"{SCHEMA}.{STORE_NAME}"

w = WorkspaceClient()
print(f"Connected to: {w.config.host}")
print(f"Schema: {SCHEMA}")
print(f"Endpoint: {ENDPOINT_NAME}")
print(f"Index: {INDEX_NAME}")

# --- Load data ---
items = []
with open("data/items.csv") as f:
    for row in csv.DictReader(f):
        items.append({
            "item_id": row["item_id"],
            "embedding": json.loads(row["embedding"]),
            "label": row["label"],
        })

queries = []
with open("data/queries.csv") as f:
    for row in csv.DictReader(f):
        queries.append({
            "query_id": row["query_id"],
            "embedding": json.loads(row["embedding"]),
        })

print(f"Loaded {len(items)} items and {len(queries)} queries")

# --- Create endpoint (or reuse) ---
try:
    ep = w.vector_search_endpoints.get_endpoint(ENDPOINT_NAME)
    print(f"Endpoint already exists: {ep.name} / {ep.endpoint_status}")
except Exception:
    print(f"Creating endpoint: {ENDPOINT_NAME}")
    ep = w.vector_search_endpoints.create_endpoint_and_wait(
        name=ENDPOINT_NAME,
        endpoint_type=vectorsearch.EndpointType.STANDARD,
        timeout=timedelta(minutes=20),
    )
    print(f"Endpoint created: {ep.name}")

# --- Create direct-access index ---
schema_json = json.dumps({
    "item_id": "string",
    "embedding": "array<float>",
    "label": "string",
})

try:
    idx = w.vector_search_indexes.get_index(INDEX_NAME)
    print(f"Index already exists: {idx.name}")
except Exception:
    print(f"Creating index: {INDEX_NAME}")
    idx = w.vector_search_indexes.create_index(
        name=INDEX_NAME,
        endpoint_name=ENDPOINT_NAME,
        primary_key="item_id",
        index_type=vectorsearch.VectorIndexType.DIRECT_ACCESS,
        direct_access_index_spec=vectorsearch.DirectAccessVectorIndexSpec(
            embedding_vector_columns=[
                vectorsearch.EmbeddingVectorColumn(
                    name="embedding",
                    embedding_dimension=16,
                )
            ],
            schema_json=schema_json,
        ),
    )
    print(f"Index created: {idx.name}")

# --- Wait for index to be ready ---
print("Waiting for index to be ready...")
for attempt in range(60):
    idx = w.vector_search_indexes.get_index(INDEX_NAME)
    if idx.status and idx.status.ready:
        print(f"  Index is ready after {attempt+1} checks")
        break
    msg = idx.status.message if idx.status else "unknown"
    print(f"  [{attempt+1}] Not ready yet: {msg}")
    time.sleep(10)
else:
    raise RuntimeError("Index did not become ready in time")

# --- Upsert items ---
print("Upserting items...")
BATCH = 100
for i in range(0, len(items), BATCH):
    batch = items[i:i+BATCH]
    result = w.vector_search_indexes.upsert_data_vector_index(
        index_name=INDEX_NAME,
        inputs_json=json.dumps(batch),
    )
    print(f"  Upserted batch {i//BATCH + 1}: {result}")

# Wait a moment for indexing to propagate
print("Waiting for index to settle...")
time.sleep(10)

# --- Query for each query vector ---
print("Querying for top-5 neighbors...")
neighbors = {}
for q in queries:
    resp = w.vector_search_indexes.query_index(
        index_name=INDEX_NAME,
        columns=["item_id"],
        query_vector=q["embedding"],
        num_results=5,
        query_type="ANN",
    )
    item_ids = []
    if resp.result and resp.result.data_array:
        for row in resp.result.data_array:
            item_ids.append(row[0])
    neighbors[q["query_id"]] = item_ids
    print(f"  {q['query_id']}: {item_ids}")

# --- Write submission ---
os.makedirs("submission", exist_ok=True)
output = {
    "store": STORE_NAME,
    "neighbors": neighbors,
}
with open("submission/answers.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\nDone. Written submission/answers.json")
print(f"Store: {STORE_NAME}")
print(f"Total queries answered: {len(neighbors)}")
