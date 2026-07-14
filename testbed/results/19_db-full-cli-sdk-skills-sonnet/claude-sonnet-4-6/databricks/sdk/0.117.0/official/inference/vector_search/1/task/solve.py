import os
import json
import csv
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import (
    EndpointStatusState, EndpointType,
    DirectAccessVectorIndexSpec, EmbeddingVectorColumn, VectorIndexType
)

w = WorkspaceClient()

schema = os.environ["MLPAB_DATABRICKS_SCHEMA"]   # workspace.mlpab321bfe
prefix = os.environ["MLPAB_DATABRICKS_PREFIX"]    # mlpab321bfe

store_name = "items9110bd"
endpoint_name = f"{prefix}_{store_name}"
index_name = f"{schema}.{store_name}"

print(f"Schema: {schema}")
print(f"Endpoint: {endpoint_name}")
print(f"Index: {index_name}")

# Load items
items = []
with open("data/items.csv") as f:
    for row in csv.DictReader(f):
        items.append({
            "item_id": row["item_id"],
            "embedding": json.loads(row["embedding"])
        })

# Load queries
queries = []
with open("data/queries.csv") as f:
    for row in csv.DictReader(f):
        queries.append({
            "query_id": row["query_id"],
            "embedding": json.loads(row["embedding"])
        })

print(f"Loaded {len(items)} items, {len(queries)} queries")

# --- Endpoint ---
try:
    ep = w.vector_search_endpoints.get_endpoint(endpoint_name)
    print(f"Endpoint already exists, state={ep.endpoint_status.state}")
except Exception:
    ep = None

if ep is None:
    print(f"Creating endpoint {endpoint_name}...")
    w.vector_search_endpoints.create_endpoint(
        name=endpoint_name,
        endpoint_type=EndpointType.STANDARD
    )

# Poll until endpoint is online
print("Waiting for endpoint to be ONLINE...")
for attempt in range(120):
    ep = w.vector_search_endpoints.get_endpoint(endpoint_name)
    state = ep.endpoint_status.state if ep.endpoint_status else None
    print(f"  attempt {attempt+1}: state={state}")
    if state == EndpointStatusState.ONLINE:
        break
    time.sleep(15)
else:
    raise RuntimeError("Endpoint did not come online in time")

print("Endpoint is ONLINE")

# --- Index ---
index_exists = False
try:
    idx = w.vector_search_indexes.get_index(index_name)
    print(f"Index already exists: {idx.status}")
    index_exists = True
except Exception:
    index_exists = False

if not index_exists:
    print(f"Creating Direct Access index {index_name}...")
    schema_json = json.dumps({
        "item_id": "string",
        "embedding": "array<float>"
    })
    w.vector_search_indexes.create_index(
        name=index_name,
        endpoint_name=endpoint_name,
        primary_key="item_id",
        index_type=VectorIndexType.DIRECT_ACCESS,
        direct_access_index_spec=DirectAccessVectorIndexSpec(
            embedding_vector_columns=[
                EmbeddingVectorColumn(name="embedding", embedding_dimension=16)
            ],
            schema_json=schema_json
        )
    )
    print("Index created, polling until ready...")

# Poll until index is ready
for attempt in range(60):
    idx = w.vector_search_indexes.get_index(index_name)
    ready = idx.status.ready if idx.status else False
    msg = idx.status.message if idx.status else "no status"
    print(f"  attempt {attempt+1}: ready={ready}, msg={msg}")
    if ready:
        break
    time.sleep(15)
else:
    raise RuntimeError("Index did not become ready in time")

# --- Upsert items ---
print("Upserting items...")
batch_size = 50
for i in range(0, len(items), batch_size):
    batch = items[i:i + batch_size]
    w.vector_search_indexes.upsert_data_vector_index(
        index_name=index_name,
        inputs_json=json.dumps(batch)
    )
    print(f"  upserted {min(i + batch_size, len(items))}/{len(items)}")

print("Waiting for indexing to settle...")
time.sleep(20)

# --- Query ---
print("Querying for top-5 neighbors...")
neighbors = {}

for q in queries:
    results = w.vector_search_indexes.query_index(
        index_name=index_name,
        columns=["item_id"],
        query_vector=q["embedding"],
        num_results=5
    )
    item_ids = [row[0] for row in results.result.data_array]
    neighbors[q["query_id"]] = item_ids
    print(f"  {q['query_id']}: {item_ids}")

# --- Write submission ---
os.makedirs("submission", exist_ok=True)
answer = {"store": store_name, "neighbors": neighbors}
with open("submission/answers.json", "w") as f:
    json.dump(answer, f, indent=2)

print(f"\nDone. Written submission/answers.json with {len(neighbors)} queries.")
