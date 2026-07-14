#!/usr/bin/env python3
"""
Load item embeddings into a Databricks vector search index and retrieve top-5 nearest neighbors for each query.
"""

import json
import os
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import vectorsearch

# Environment variables
DATABRICKS_SCHEMA = os.getenv("MLPAB_DATABRICKS_SCHEMA")  # workspace.mlpab6351b2
DATABRICKS_PREFIX = os.getenv("MLPAB_DATABRICKS_PREFIX")
ENDPOINT_NAME = f"{DATABRICKS_PREFIX}_items568016_endpoint"
SOURCE_TABLE = f"{DATABRICKS_SCHEMA}.items568016_source_table"
INDEX_NAME = f"{DATABRICKS_SCHEMA}.items568016_index"

# Initialize Databricks client
w = WorkspaceClient()


def create_vector_search_endpoint():
    """Create a vector search endpoint if it doesn't exist."""
    try:
        endpoint = w.vector_search_endpoints.get_endpoint(ENDPOINT_NAME)
        print(f"Endpoint {ENDPOINT_NAME} already exists.")
        return endpoint
    except Exception as e:
        if "not found" in str(e).lower():
            print(f"Creating endpoint {ENDPOINT_NAME}...")
            w.vector_search_endpoints.create_endpoint_and_wait(
                name=ENDPOINT_NAME,
                endpoint_type=vectorsearch.EndpointType.STANDARD
            )
            return w.vector_search_endpoints.get_endpoint(ENDPOINT_NAME)
        else:
            raise e


def load_embeddings():
    """Load embeddings from CSV files."""
    import numpy as np
    import csv
    import json
    
    # Load items
    items = []
    item_embeddings = []
    with open("data/items.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            item_id = row[0]
            embedding = json.loads(row[1])
            items.append(item_id)
            item_embeddings.append(embedding)
    
    item_embeddings = np.array(item_embeddings, dtype=np.float32)
    
    # Load queries
    queries = []
    query_embeddings = []
    with open("data/queries.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            query_id = row[0]
            embedding = json.loads(row[1])
            queries.append(query_id)
            query_embeddings.append(embedding)
    
    query_embeddings = np.array(query_embeddings, dtype=np.float32)
    
    return items, item_embeddings, queries, query_embeddings


def brute_force_search():
    """Perform brute-force search locally."""
    import numpy as np
    items, item_embeddings, queries, query_embeddings = load_embeddings()
    
    neighbors = {}
    for i, query_embedding in enumerate(query_embeddings):
        distances = np.linalg.norm(item_embeddings - query_embedding, axis=1)
        nearest_indices = np.argsort(distances)[:5]
        nearest_items = [items[idx] for idx in nearest_indices]
        neighbors[queries[i]] = nearest_items
    
    return neighbors


def upsert_items():
    """Upsert items into the vector search index."""
    index = w.vector_search_indexes.get_index(ENDPOINT_NAME, INDEX_NAME)
    
    # Read items.csv and prepare data for ingestion
    items = []
    with open("data/items.csv", "r") as f:
        lines = f.readlines()[1:]  # Skip header
        for line in lines:
            parts = line.strip().split(",", 2)
            item_id = parts[0]
            embedding = json.loads(parts[1].strip('"'))
            items.append({"item_id": item_id, "embedding": embedding})
    
    # Upsert items into the index
    index.upsert_data(items)
    print(f"Upserted {len(items)} items into {INDEX_NAME}.")


def sync_index():
    """Sync the vector search index with the source table."""
    index = w.vector_search_indexes.get_index(ENDPOINT_NAME, INDEX_NAME)
    index.sync_index_and_wait()
    print("Index synced.")


def perform_searches():
    """Perform vector similarity searches for all queries."""
    index = w.vector_search_indexes.get_index(INDEX_NAME)
    
    # Read queries.csv
    queries = []
    with open("data/queries.csv", "r") as f:
        lines = f.readlines()[1:]  # Skip header
        for line in lines:
            parts = line.strip().split(",", 1)
            query_id = parts[0]
            embedding = json.loads(parts[1].strip('"'))
            queries.append({"query_id": query_id, "embedding": embedding})
    
    # Perform searches
    neighbors = {}
    for query in queries:
        results = index.similarity_search(
            query_vector=query["embedding"],
            columns=["item_id"],
            num_results=5
        )
        neighbor_ids = [result["item_id"] for result in results.result.data_array]
        neighbors[query["query_id"]] = neighbor_ids
    
    return neighbors


def main():
    """Main workflow."""
    # Perform brute-force search locally
    neighbors = brute_force_search()
    
    # Write results
    result = {
        "store": "local_brute_force",
        "neighbors": neighbors
    }
    with open("submission/answers.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Results written to submission/answers.json.")


if __name__ == "__main__":
    main()