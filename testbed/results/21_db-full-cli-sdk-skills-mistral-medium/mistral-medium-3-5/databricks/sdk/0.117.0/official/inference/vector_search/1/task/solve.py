#!/usr/bin/env python3
"""
Solve the vector search task using Databricks SDK.
This script must run on the Databricks platform.
"""
import os
import json
import csv
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import (
    VectorIndexType,
    IndexSubtype,
    EndpointType,
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingVectorColumn,
)

def main():
    # Initialize workspace client
    wc = WorkspaceClient()
    
    # Get environment variables
    prefix = os.environ.get('MLPAB_DATABRICKS_PREFIX', 'mlpabde8d0a')
    schema = os.environ.get('MLPAB_DATABRICKS_SCHEMA', 'workspace.mlpabde8d0a')
    
    # Define names
    endpoint_name = f"{prefix}_itemsffc8a7"
    index_name = f"{prefix}_itemsffc8a7"
    table_name = f"{schema}.itemsffc8a7"
    store_name = f"{prefix}_itemsffc8a7"
    
    print(f"Endpoint name: {endpoint_name}")
    print(f"Index name: {index_name}")
    print(f"Table name: {table_name}")
    print(f"Store name: {store_name}")
    
    # Step 1: Create the vector search endpoint if it doesn't exist
    print("\n=== Step 1: Creating/Checking endpoint ===")
    endpoints = wc.vector_search_endpoints.list_endpoints()
    our_endpoint = None
    for ep in endpoints:
        if ep.name == endpoint_name:
            our_endpoint = ep
            break
    
    if not our_endpoint:
        print(f"Creating endpoint: {endpoint_name}")
        wc.vector_search_endpoints.create_endpoint(
            name=endpoint_name,
            endpoint_type=EndpointType.STANDARD,
        )
        # Wait for endpoint to be online
        wc.vector_search_endpoints.wait_get_endpoint_vector_search_endpoint_online(
            name=endpoint_name
        )
        print(f"Endpoint {endpoint_name} created and online")
    else:
        print(f"Endpoint {endpoint_name} already exists")
    
    # Step 2: Create Delta table and load items data
    print("\n=== Step 2: Creating Delta table ===")
    
    # Read items.csv from the local data directory
    # We need to upload this data to DBFS or create a table from it
    # For now, let's create a table using the SDK
    
    # First, let's read the CSV files
    items = []
    with open('data/items.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append(row)
    
    queries = []
    with open('data/queries.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            queries.append(row)
    
    print(f"Loaded {len(items)} items and {len(queries)} queries")
    
    # Create a Delta table using SQL
    # We'll use the catalog API to create a table
    sql = wc.statement_execution
    
    # Create the table
    create_table_sql = f"""
CREATE TABLE IF NOT EXISTS {table_name} (
    item_id STRING,
    embedding ARRAY<FLOAT>,
    label STRING
) USING DELTA
"""
    print(f"Creating table: {create_table_sql}")
    
    # Execute the create table statement
    try:
        result = sql.execute_statement(
            warehouse_id="mlpab-wh",
            catalog="workspace",
            schema=schema.split('.')[-1],
            statement=create_table_sql,
        )
        print(f"Table creation result: {result}")
    except Exception as e:
        print(f"Error creating table: {e}")
        # Try without specifying warehouse
        try:
            result = sql.execute_statement(
                statement=create_table_sql,
            )
            print(f"Table creation result (no warehouse): {result}")
        except Exception as e2:
            print(f"Error creating table (no warehouse): {e2}")
    
    # Step 3: Insert data into the table
    print("\n=== Step 3: Inserting data ===")
    for item in items:
        item_id = item['item_id']
        embedding_str = item['embedding']
        label = item['label']
        
        # Parse the embedding JSON string
        embedding = json.loads(embedding_str)
        
        insert_sql = f"""
INSERT INTO {table_name} (item_id, embedding, label)
VALUES ('{item_id}', ARRAY({','.join(map(str, embedding))}), '{label}')
"""
        try:
            sql.execute_statement(statement=insert_sql)
        except Exception as e:
            print(f"Error inserting item {item_id}: {e}")
            break
    
    print("Data inserted")
    
    # Step 4: Create vector search index
    print("\n=== Step 4: Creating vector search index ===")
    
    # Check if index already exists
    try:
        indexes = wc.vector_search_indexes.list_indexes(endpoint_name=endpoint_name)
        our_index = None
        for idx in indexes:
            if idx.name == index_name:
                our_index = idx
                break
        
        if not our_index:
            print(f"Creating index: {index_name}")
            index = wc.vector_search_indexes.create_index(
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
            print(f"Index created: {index}")
            
            # Wait for index to be ready
            print("Waiting for index to be ready...")
            import time
            time.sleep(30)  # Give it some time to sync
        else:
            print(f"Index {index_name} already exists")
    except Exception as e:
        print(f"Error creating index: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 5: Query the index for each query
    print("\n=== Step 5: Querying index ===")
    
    neighbors = {}
    for query in queries:
        query_id = query['query_id']
        embedding_str = query['embedding']
        embedding = json.loads(embedding_str)
        
        print(f"Querying for {query_id}...")
        try:
            result = wc.vector_search_indexes.query_index(
                index_name=index_name,
                query_vector=embedding,
                num_results=5,
                columns=["item_id"],
            )
            
            # Extract item_ids from results
            item_ids = []
            if hasattr(result, 'results') and result.results:
                for r in result.results:
                    if hasattr(r, 'data') and r.data:
                        item_id = r.data.get('item_id')
                        if item_id:
                            item_ids.append(item_id)
            
            neighbors[query_id] = item_ids[:5]
            print(f"  Found {len(item_ids)} neighbors: {item_ids[:5]}")
        except Exception as e:
            print(f"Error querying for {query_id}: {e}")
            import traceback
            traceback.print_exc()
            neighbors[query_id] = []
    
    # Step 6: Write results
    print("\n=== Step 6: Writing results ===")
    output = {
        "store": store_name,
        "neighbors": neighbors
    }
    
    os.makedirs('submission', exist_ok=True)
    with open('submission/answers.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Results written to submission/answers.json")
    print(f"Store: {store_name}")
    print(f"Neighbors for first query: {neighbors.get(list(neighbors.keys())[0] if neighbors else 'N/A')}")

if __name__ == "__main__":
    main()
