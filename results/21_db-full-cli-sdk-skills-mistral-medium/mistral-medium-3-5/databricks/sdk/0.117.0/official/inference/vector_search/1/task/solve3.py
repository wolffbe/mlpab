#!/usr/bin/env python3
"""
Script to solve the vector search task using DELTA_SYNC index.
Since there's already a table workspace.mlpabde8d0a.itemsffc8a7, we'll use DELTA_SYNC.
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
    EndpointType,
)

# Environment variables
MLPAB_DATABRICKS_SCHEMA = os.environ.get("MLPAB_DATABRICKS_SCHEMA", "workspace.mlpabde8d0a")
MLPAB_DATABRICKS_PREFIX = os.environ.get("MLPAB_DATABRICKS_PREFIX", "mlpabde8d0a")

# Constants - the store/index name should be exactly "itemsffc8a7"
STORE_NAME = "itemsffc8a7"
ENDPOINT_NAME = f"{MLPAB_DATABRICKS_PREFIX}_{STORE_NAME}"
# Table name
TABLE_NAME = f"{MLPAB_DATABRICKS_SCHEMA}.{STORE_NAME}"
# Index name must be in format <catalog>.<schema>.<table>
INDEX_NAME = f"{MLPAB_DATABRICKS_SCHEMA}.{STORE_NAME}_vec_idx"

def main():
    # Initialize Databricks client
    wc = WorkspaceClient()
    
    print(f"Store name: {STORE_NAME}")
    print(f"Endpoint name: {ENDPOINT_NAME}")
    print(f"Table name: {TABLE_NAME}")
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
    
    # Step 2: Check if table exists and has data
    print("\n=== Step 2: Checking table ===")
    try:
        sql = wc.statement_execution
        warehouse_id = "8a93fc195da2ceb1"  # mlpab-grader warehouse
        
        # Check if table exists
        tables = list(wc.tables.list(catalog_name='workspace', schema_name=MLPAB_DATABRICKS_SCHEMA.split('.')[-1]))
        table_exists = any(t.full_name == TABLE_NAME for t in tables)
        
        if not table_exists:
            print(f"Table {TABLE_NAME} does not exist, creating it...")
            # Create the table with change data feed enabled
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                item_id STRING,
                embedding ARRAY<FLOAT>,
                label STRING
            ) USING DELTA
            TBLPROPERTIES (delta.enableChangeDataFeed = true)
            """
            sql.execute_statement(statement=create_table_sql, warehouse_id=warehouse_id)
            print(f"Table {TABLE_NAME} created")
        else:
            print(f"Table {TABLE_NAME} exists")
            # Ensure change data feed is enabled
            alter_table_sql = f"""
            ALTER TABLE {TABLE_NAME} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
            """
            try:
                sql.execute_statement(statement=alter_table_sql, warehouse_id=warehouse_id)
                print(f"Enabled change data feed on {TABLE_NAME}")
            except Exception as e:
                print(f"Change data feed might already be enabled: {e}")
        
        # Check if table has data
        count_result = sql.execute_statement(statement=f"SELECT COUNT(*) as cnt FROM {TABLE_NAME}", warehouse_id=warehouse_id)
        row_count = count_result.result.data_array[0][0] if count_result.result and count_result.result.data_array else 0
        print(f"Table has {row_count} rows")
        
        if row_count == 0:
            print("Table is empty, inserting data...")
            for item in items:
                item_id = item['item_id']
                embedding = item['embedding']
                label = item.get('label', '')
                
                insert_sql = f"""
                INSERT INTO {TABLE_NAME} (item_id, embedding, label)
                VALUES ('{item_id}', ARRAY({','.join(map(str, embedding))}), '{label}')
                """
                sql.execute_statement(statement=insert_sql, warehouse_id=warehouse_id)
            print("Data inserted")
        else:
            print("Table already has data")
    except Exception as e:
        print(f"Error with table: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 3: Create DELTA_SYNC index
    print("\n=== Step 3: Creating index ===")
    try:
        indexes = list(wc.vector_search_indexes.list_indexes(endpoint_name=ENDPOINT_NAME))
        index_names = [i.name for i in indexes]
        
        if INDEX_NAME in index_names:
            print(f"Index {INDEX_NAME} already exists")
        else:
            print(f"Creating index: {INDEX_NAME}")
            
            from databricks.sdk.service.vectorsearch import EmbeddingVectorColumn, PipelineType
            
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
            # Give it a moment
            time.sleep(5)
            
            # Wait for index to sync - DELTA_SYNC can take a while
            print("Waiting for index to sync...")
            time.sleep(60)  # Wait 60 seconds for initial sync
    except Exception as e:
        print(f"Error creating index: {e}")
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
        
        # Retry logic for index not ready
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
                
                # Extract item_ids from result
                item_ids = []
                if result.result and result.result.data_array:
                    for row in result.result.data_array:
                        if row and len(row) > 0:
                            item_ids.append(row[0])
                
                neighbors[query_id] = item_ids
                print(f"    {query_id}: {item_ids}")
                break  # Success, exit retry loop
            except Exception as e:
                if "not ready" in str(e) and attempt < max_retries - 1:
                    print(f"    Index not ready, retrying in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                else:
                    print(f"Error querying for {query_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    neighbors[query_id] = []
                    break
    
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
