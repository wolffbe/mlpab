# Databricks notebook source
# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import (
    EndpointType, VectorIndexType,
    DirectAccessVectorIndexSpec, EmbeddingVectorColumn
)
import json
import time
import csv

w = WorkspaceClient()

ENDPOINT_NAME = "mlpab3fd1e9_items9110bd"
INDEX_NAME = "workspace.mlpab3fd1e9.items9110bd"
VOLUME_PATH = "/Volumes/workspace/mlpab3fd1e9/mlpab3fd1e9data"
RESULTS_PATH = f"{VOLUME_PATH}/answers.json"

print(f"Endpoint: {ENDPOINT_NAME}")
print(f"Index: {INDEX_NAME}")

# COMMAND ----------

# Read items from volume
items = []
with open(f"{VOLUME_PATH}/items.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        items.append({
            "item_id": row["item_id"],
            "embedding": json.loads(row["embedding"])
        })
print(f"Loaded {len(items)} items")

# Read queries from volume
queries = []
with open(f"{VOLUME_PATH}/queries.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        queries.append({
            "query_id": row["query_id"],
            "embedding": json.loads(row["embedding"])
        })
print(f"Loaded {len(queries)} queries")

# COMMAND ----------

# Create or wait for endpoint to be ONLINE
print("Checking endpoint status...")
try:
    ep = w.vector_search_endpoints.get_endpoint(ENDPOINT_NAME)
    state_val = ep.endpoint_status.state.value if ep.endpoint_status and ep.endpoint_status.state else "UNKNOWN"
    print(f"Endpoint exists, state: {state_val}")
except Exception as e:
    print(f"Endpoint not found, creating: {e}")
    w.vector_search_endpoints.create_endpoint(
        name=ENDPOINT_NAME,
        endpoint_type=EndpointType.STANDARD
    )
    print("Endpoint creation initiated")

# Wait for endpoint to be ONLINE (max 40 minutes)
print("Waiting for endpoint to be ONLINE...")
for i in range(80):
    ep = w.vector_search_endpoints.get_endpoint(ENDPOINT_NAME)
    if ep.endpoint_status and ep.endpoint_status.state:
        state_val = ep.endpoint_status.state.value
    else:
        state_val = "UNKNOWN"
    print(f"  [{i}] Endpoint state: {state_val}")
    if state_val == "ONLINE":
        print("Endpoint is ONLINE!")
        break
    time.sleep(30)
else:
    raise Exception("Endpoint did not become ONLINE in time")

# COMMAND ----------

# Create or get index
index_exists = False
try:
    idx = w.vector_search_indexes.get_index(INDEX_NAME)
    print(f"Index already exists, ready: {idx.status.ready if idx.status else 'N/A'}")
    index_exists = True
except Exception as e:
    print(f"Index not found, creating: {e}")

if not index_exists:
    w.vector_search_indexes.create_index(
        name=INDEX_NAME,
        endpoint_name=ENDPOINT_NAME,
        primary_key="item_id",
        index_type=VectorIndexType.DIRECT_ACCESS,
        direct_access_index_spec=DirectAccessVectorIndexSpec(
            embedding_vector_columns=[
                EmbeddingVectorColumn(
                    name="embedding",
                    embedding_dimension=16
                )
            ],
            schema_json=json.dumps({
                "item_id": "string",
                "embedding": "array<float>"
            })
        )
    )
    print("Index creation initiated")

# Wait for index to be ready (max 10 minutes)
print("Waiting for index to be ready...")
for i in range(60):
    idx = w.vector_search_indexes.get_index(INDEX_NAME)
    ready = idx.status.ready if idx.status else False
    msg = idx.status.message if idx.status else "N/A"
    print(f"  [{i}] Index ready: {ready}, message: {msg}")
    if ready:
        print("Index is ready!")
        break
    time.sleep(10)
else:
    print("WARNING: Index may not be ready, proceeding anyway")

# COMMAND ----------

# Upsert items in batches of 100
# Note: pass index_name as positional arg (SDK version compatibility)
BATCH_SIZE = 100
print(f"Upserting {len(items)} items in batches of {BATCH_SIZE}...")
for i in range(0, len(items), BATCH_SIZE):
    batch = items[i:i + BATCH_SIZE]
    result = w.vector_search_indexes.upsert_data_vector_index(INDEX_NAME, json.dumps(batch))
    print(f"  Upserted items {i}-{i+len(batch)-1}")

# Give the index time to process the upserts
print("Waiting for index to process upserts...")
time.sleep(15)

# COMMAND ----------

# Query all queries for top-5 nearest neighbors
# Note: pass index_name as positional arg (SDK version compatibility)
print(f"Querying {len(queries)} queries...")
neighbors = {}
for q in queries:
    try:
        result = w.vector_search_indexes.query_index(
            INDEX_NAME,
            columns=["item_id"],
            query_vector=q["embedding"],
            num_results=5
        )
        if result.result and result.result.data_array:
            item_ids = [row[0] for row in result.result.data_array]
        else:
            item_ids = []
            print(f"  WARNING: No results for query {q['query_id']}")
        neighbors[q["query_id"]] = item_ids
        print(f"  Query {q['query_id']}: {item_ids}")
    except Exception as e:
        print(f"  ERROR for query {q['query_id']}: {e}")
        neighbors[q["query_id"]] = []

# COMMAND ----------

# Write results to volume
answers = {
    "store": "workspace.mlpab3fd1e9.items9110bd",
    "neighbors": neighbors
}

with open(RESULTS_PATH, "w") as f:
    json.dump(answers, f, indent=2)

print(f"\nResults written to {RESULTS_PATH}")
print(json.dumps(answers, indent=2))
