#!/usr/bin/env python3
"""
Simplified script to solve the vector search task using DIRECT_ACCESS index.
"""
import os
import csv
import json
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import (
    VectorIndexType,
    IndexSubtype,
    DirectAccessVectorIndexSpec,
    EmbeddingVectorColumn,
    EndpointType,
)

# Environment variables
MLPAB_DATABRICKS_SCHEMA = os.environ.get("MLPAB_DATABRICKS_SCHEMA", "workspace.mlpabde8d0a")
MLPAB_DATABRICKS_PREFIX = os.environ.get("MLPAB_DATABRICKS_PREFIX", "mlpabde8d0a")

# Constants - the store/index name should be exactly "itemsffc8a7"
STORE_NAME = "itemsffc8a7"
ENDPOINT_NAME = f"{MLPAB_DATABRICKS_PREFIX}_{STORE_NAME}"
# Index name must be in format <catalog>.<schema>.<table>
# The schema is workspace.mlpabde8d0a, so the full index name is workspace.mlpabde8d0a.itemsffc8a7_vec_idx
# We use a unique name to avoid conflicts
INDEX_NAME = f"{MLPAB_DATABRICKS_SCHEMA}.{STORE_NAME}_vec_idx"

def main():
    # Initialize Databricks client
    wc = WorkspaceClient()
    
    print(f"Store name: {STORE_NAME}")
    print(f"Endpoint name: {ENDPOINT_NAME}")
    print(f"Index name: {INDEX_NAME}")
    
    # Load items
    print("\n=== Loading items ===")
    items = []
    with open('data/items.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['embedding'] = json.loads(row['embedding'])
            items.append(row)
    print(f"Loaded {len(items)} items")
    
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
        # Wait for endpoint to be ready
        print("Waiting for endpoint to be ready...")
        wc.vector_search_endpoints.wait_get_endpoint_vector_search_endpoint_online(
            name=ENDPOINT_NAME,
            timeout=300
        )
        print("Endpoint is ready")
    else:
        print(f"Endpoint {ENDPOINT_NAME} already exists")
    
    # Step 2: Create DIRECT_ACCESS index
    print("\n=== Step 2: Creating index ===")
    try:
        indexes = list(wc.vector_search_indexes.list_indexes(endpoint_name=ENDPOINT_NAME))
        index_names = [i.name for i in indexes]
        
        if INDEX_NAME in index_names:
            print(f"Index {INDEX_NAME} already exists")
        else:
            print(f"Creating index: {INDEX_NAME}")
            
            # Define schema for the index
            schema_json = json.dumps({
                "columns": [
                    {"name": "item_id", "type": "string"},
                    {"name": "embedding", "type": "array<float>"},
                    {"name": "label", "type": "string"}
                ]
            })
            
            spec = DirectAccessVectorIndexSpec(
                embedding_vector_columns=[
                    EmbeddingVectorColumn(
                        name="embedding",
                        embedding_dimension=16
                    )
                ],
                schema_json=schema_json
            )
            
            wc.vector_search_indexes.create_index(
                name=INDEX_NAME,
                endpoint_name=ENDPOINT_NAME,
                primary_key="item_id",
                index_type=VectorIndexType.DIRECT_ACCESS,
                index_subtype=IndexSubtype.VECTOR,
                direct_access_index_spec=spec
            )
            print(f"Index {INDEX_NAME} created")
            # Give it a moment
            time.sleep(5)
    except Exception as e:
        print(f"Error creating index: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: Upsert items
    print("\n=== Step 3: Upserting items ===")
    try:
        # Prepare data for upsert
        data_to_upsert = []
        for item in items:
            data_to_upsert.append({
                "item_id": item["item_id"],
                "embedding": item["embedding"],
                "label": item.get("label", "")
            })
        
        # Upsert in batches
        batch_size = 100
        for i in range(0, len(data_to_upsert), batch_size):
            batch = data_to_upsert[i:i + batch_size]
            batch_json = json.dumps(batch)
            print(f"  Upserting batch {i//batch_size + 1}/{(len(data_to_upsert) + batch_size - 1)//batch_size}")
            wc.vector_search_indexes.upsert_data_vector_index(
                index_name=INDEX_NAME,
                inputs_json=batch_json
            )
        print("All items upserted")
        
        # Wait for data to be indexed
        print("Waiting for data to be indexed...")
        time.sleep(15)
    except Exception as e:
        print(f"Error upserting items: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Query for each query
    print("\n=== Step 4: Querying ===")
    neighbors = {}
    for query in queries:
        query_id = query["query_id"]
        query_vector = query["embedding"]
        print(f"  Querying for {query_id}...")
        
        try:
            result = wc.vector_search_indexes.query_index(
                index_name=INDEX_NAME,
                columns=["item_id"],
                query_vector=query_vector,
                num_results=5,
                query_type="ANN"
            )
            
            # Extract item_ids from result
            item_ids = []
            if result.result and result.result.data_array:
                for row in result.result.data_array:
                    if row and len(row) > 0:
                        item_ids.append(row[0])
            
            neighbors[query_id] = item_ids
            print(f"    {query_id}: {item_ids}")
        except Exception as e:
            print(f"Error querying for {query_id}: {e}")
            import traceback
            traceback.print_exc()
            neighbors[query_id] = []
    
    # Step 5: Write results
    print("\n=== Step 5: Writing results ===")
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
