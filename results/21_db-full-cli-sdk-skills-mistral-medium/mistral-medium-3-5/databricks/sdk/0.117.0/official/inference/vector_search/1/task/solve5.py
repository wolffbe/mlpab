#!/usr/bin/env python3
"""
Use the existing table with DELTA_SYNC index.
The table workspace.mlpabde8d0a.itemsffc8a7 already exists with 300 rows and change data feed enabled.
"""
import os
import csv
import json
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import (
    VectorIndexType,
    IndexSubtype,
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingVectorColumn,
    PipelineType,
    EndpointType,
)

# Environment variables
MLPAB_DATABRICKS_SCHEMA = os.environ.get("MLPAB_DATABRICKS_SCHEMA", "workspace.mlpabde8d0a")
MLPAB_DATABRICKS_PREFIX = os.environ.get("MLPAB_DATABRICKS_PREFIX", "mlpabde8d0a")

# Constants
STORE_NAME = "itemsffc8a7"
ENDPOINT_NAME = f"{MLPAB_DATABRICKS_PREFIX}_{STORE_NAME}"
TABLE_NAME = f"{MLPAB_DATABRICKS_SCHEMA}.{STORE_NAME}"
INDEX_NAME = f"{MLPAB_DATABRICKS_SCHEMA}.{STORE_NAME}_vec_idx"

def main():
    wc = WorkspaceClient()
    
    print(f"Store name: {STORE_NAME}")
    print(f"Endpoint name: {ENDPOINT_NAME}")
    print(f"Table name: {TABLE_NAME}")
    print(f"Index name: {INDEX_NAME}")
    
    # Load queries
    print("\n=== Loading queries ===")
    queries = []
    with open('data/queries.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['embedding'] = json.loads(row['embedding'])
            queries.append(row)
    print(f"Loaded {len(queries)} queries")
    
    # Step 1: Ensure endpoint exists
    print("\n=== Step 1: Checking endpoint ===")
    endpoints = list(wc.vector_search_endpoints.list_endpoints())
    endpoint_names = [e.name for e in endpoints]
    
    if ENDPOINT_NAME not in endpoint_names:
        print(f"Creating endpoint: {ENDPOINT_NAME}")
        wc.vector_search_endpoints.create_endpoint(
            name=ENDPOINT_NAME,
            endpoint_type=EndpointType.STANDARD,
        )
        print("Waiting for endpoint to be ready...")
        wc.vector_search_endpoints.wait_get_endpoint_vector_search_endpoint_online(
            name=ENDPOINT_NAME,
            timeout=300
        )
        print("Endpoint is ready")
    else:
        print(f"Endpoint {ENDPOINT_NAME} already exists")
    
    # Step 2: Create DELTA_SYNC index
    print("\n=== Step 2: Creating index ===")
    try:
        indexes = list(wc.vector_search_indexes.list_indexes(endpoint_name=ENDPOINT_NAME))
        index_names = [i.name for i in indexes]
        
        if INDEX_NAME in index_names:
            print(f"Index {INDEX_NAME} already exists")
        else:
            print(f"Creating index: {INDEX_NAME}")
            
            spec = DeltaSyncVectorIndexSpecRequest(
                source_table=TABLE_NAME,
                embedding_vector_columns=[
                    EmbeddingVectorColumn(name="embedding", embedding_dimension=16)
                ],
                pipeline_type=PipelineType.TRIGGERED
            )
            
            wc.vector_search_indexes.create_index(
                name=INDEX_NAME,
                endpoint_name=ENDPOINT_NAME,
                primary_key="item_id",
                index_type=VectorIndexType.DELTA_SYNC,
                index_subtype=IndexSubtype.HYBRID,
                delta_sync_index_spec=spec
            )
            print(f"Index {INDEX_NAME} created")
            time.sleep(5)
            
            # Wait for index to sync - this can take a while
            print("Waiting for index to sync...")
            # Check index status periodically
            max_wait = 300  # 5 minutes
            wait_interval = 10
            for i in range(max_wait // wait_interval):
                try:
                    idx = wc.vector_search_indexes.get_index(index_name=INDEX_NAME)
                    # If we can get the index without error, it's probably ready
                    print(f"  Index status check {i+1}: OK")
                    break
                except Exception as e:
                    if "not ready" in str(e):
                        print(f"  Index not ready yet, waiting... ({i+1}/{max_wait//wait_interval})")
                        time.sleep(wait_interval)
                    else:
                        print(f"  Error checking index: {e}")
                        time.sleep(wait_interval)
    except Exception as e:
        print(f"Error creating index: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: Query for each query
    print("\n=== Step 3: Querying ===")
    neighbors = {}
    for query in queries:
        query_id = query["query_id"]
        query_vector = query["embedding"]
        print(f"  Querying for {query_id}...")
        
        # Retry logic
        max_retries = 10
        retry_delay = 10
        for attempt in range(max_retries):
            try:
                result = wc.vector_search_indexes.query_index(
                    index_name=INDEX_NAME,
                    columns=["item_id"],
                    query_vector=query_vector,
                    num_results=5,
                    query_type="ANN"
                )
                
                item_ids = []
                if result.result and result.result.data_array:
                    for row in result.result.data_array:
                        if row and len(row) > 0:
                            item_ids.append(row[0])
                
                neighbors[query_id] = item_ids
                print(f"    {query_id}: {item_ids}")
                break
            except Exception as e:
                if "not ready" in str(e) and attempt < max_retries - 1:
                    print(f"    Index not ready, retrying in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                else:
                    print(f"Error querying for {query_id}: {e}")
                    neighbors[query_id] = []
                    break
    
    # Step 4: Write results
    print("\n=== Step 4: Writing results ===")
    os.makedirs('submission', exist_ok=True)
    output = {
        "store": STORE_NAME,
        "neighbors": neighbors
    }
    
    with open('submission/answers.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Results written to submission/answers.json")
    print(f"Store: {STORE_NAME}")
    print(f"Number of queries: {len(neighbors)}")

if __name__ == "__main__":
    main()
