#!/usr/bin/env python3
"""
Run vector search task on Databricks platform using SDK.
"""
import os
import json
import csv
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import (
    VectorIndexType,
    IndexSubtype,
    EndpointType,
    DeltaSyncVectorIndexSpecRequest,
)

def main():
    wc = WorkspaceClient()
    
    # Environment
    prefix = os.environ.get('MLPAB_DATABRICKS_PREFIX', 'mlpabde8d0a')
    schema = os.environ.get('MLPAB_DATABRICKS_SCHEMA', 'workspace.mlpabde8d0a')
    
    # Names
    endpoint_name = f"{prefix}_itemsffc8a7"
    index_name = "itemsffc8a7"  # The store/index name as per task
    table_name = f"{schema}.itemsffc8a7"
    store_name = "itemsffc8a7"  # The store name as per task
    
    print(f"Endpoint: {endpoint_name}")
    print(f"Index: {index_name}")
    print(f"Table: {table_name}")
    print(f"Store: {store_name}")
    
    # Step 1: Create endpoint
    print("\n=== Step 1: Create endpoint ===")
    endpoints = wc.vector_search_endpoints.list_endpoints()
    our_endpoint = next((ep for ep in endpoints if ep.name == endpoint_name), None)
    
    if not our_endpoint:
        print(f"Creating endpoint: {endpoint_name}")
        wc.vector_search_endpoints.create_endpoint(
            name=endpoint_name,
            endpoint_type=EndpointType.STANDARD,
        )
        # Wait for it to be online
        print("Waiting for endpoint to be online...")
        wc.vector_search_endpoints.wait_get_endpoint_vector_search_endpoint_online(
            endpoint_name=endpoint_name
        )
        print(f"Endpoint {endpoint_name} is online")
    else:
        print(f"Endpoint {endpoint_name} already exists")
    
    # Step 2: Create table and load data
    print("\n=== Step 2: Create table and load data ===")
    
    # Get warehouse
    warehouses = wc.warehouses.list()
    warehouse_id = warehouses[0].id if warehouses else None
    
    if not warehouse_id:
        print("No warehouse found!")
        return
    
    # Create table
    create_sql = f"""
CREATE TABLE IF NOT EXISTS {table_name} (
    item_id STRING,
    embedding ARRAY<FLOAT>,
    label STRING
) USING DELTA
"""
    print(f"Creating table...")
    wc.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=create_sql,
    )
    print(f"Table {table_name} created")
    
    # Load items
    items = []
    with open('data/items.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append(row)
    
    print(f"Loading {len(items)} items...")
    for item in items:
        item_id = item['item_id']
        embedding = json.loads(item['embedding'])
        label = item['label']
        
        # Build ARRAY constructor
        embedding_str = ',' + ','.join(map(str, embedding))
        insert_sql = f"""
INSERT INTO {table_name} (item_id, embedding, label)
VALUES ('{item_id}', ARRAY[{embedding_str}], '{label}')
"""
        wc.statement_execution.execute_statement(
            warehouse_id=warehouse_id,
            statement=insert_sql,
        )
    
    print(f"Loaded {len(items)} items")
    
    # Step 3: Create vector search index
    print("\n=== Step 3: Create vector search index ===")
    
    try:
        indexes = wc.vector_search_indexes.list_indexes(endpoint_name=endpoint_name)
        our_index = next((idx for idx in indexes if idx.name == index_name), None)
        
        if not our_index:
            print(f"Creating index: {index_name}")
            wc.vector_search_indexes.create_index(
                name=index_name,
                endpoint_name=endpoint_name,
                primary_key="item_id",
                index_type=VectorIndexType.DELTA_SYNC,
                delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
                    source_table_name=table_name,
                    embedding_source_columns=["embedding"],
                ),
                index_subtype=IndexSubtype.VECTOR,
            )
            print(f"Index {index_name} created")
            
            # Wait for index to sync
            print("Waiting for index to sync...")
            time.sleep(60)  # Give it time to sync
        else:
            print(f"Index {index_name} already exists")
    except Exception as e:
        print(f"Error creating index: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Query for each query
    print("\n=== Step 4: Query index ===")
    
    queries = []
    with open('data/queries.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            queries.append(row)
    
    neighbors = {}
    for query in queries:
        query_id = query['query_id']
        embedding = json.loads(query['embedding'])
        
        print(f"Querying {query_id}...")
        try:
            result = wc.vector_search_indexes.query_index(
                index_name=index_name,
                query_vector=embedding,
                num_results=5,
                columns=["item_id"],
            )
            
            item_ids = []
            if hasattr(result, 'results') and result.results:
                for r in result.results:
                    if hasattr(r, 'data') and r.data:
                        item_id = r.data.get('item_id')
                        if item_id:
                            item_ids.append(item_id)
            
            neighbors[query_id] = item_ids[:5]
            print(f"  Got {len(item_ids)} results: {item_ids[:5]}")
        except Exception as e:
            print(f"Error querying {query_id}: {e}")
            neighbors[query_id] = []
    
    # Step 5: Write results
    print("\n=== Step 5: Write results ===")
    output = {
        "store": store_name,
        "neighbors": neighbors
    }
    
    os.makedirs('submission', exist_ok=True)
    with open('submission/answers.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Results written to submission/answers.json")
    print(f"Store: {store_name}")
    print(f"Queries processed: {len(neighbors)}")

if __name__ == "__main__":
    main()
