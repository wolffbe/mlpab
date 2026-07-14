#!/usr/bin/env python3
"""
Script to:
1. Create a table in Unity Catalog with item embeddings
2. Create a vector search index on that table
3. Query the index for each query
4. Write results to submission/answers.json
"""

import os
import json
import csv
import time
import databricks.sdk
from databricks.sdk.service.vectorsearch import (
    VectorIndexType,
    IndexSubtype,
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingVectorColumn,
    EndpointType,
    PipelineType,
)

# Environment variables
MLPAB_DATABRICKS_SCHEMA = os.environ.get("MLPAB_DATABRICKS_SCHEMA", "workspace.mlpabd7bcb5")
MLPAB_DATABRICKS_PREFIX = os.environ.get("MLPAB_DATABRICKS_PREFIX", "mlpabd7bcb5")

# Store/index name - the store is the index name which must be catalog.schema.table
STORE_NAME = "itemsffc8a7"
INDEX_NAME = f"{MLPAB_DATABRICKS_SCHEMA}.{STORE_NAME}"  # Full path for index
ENDPOINT_NAME = f"{MLPAB_DATABRICKS_PREFIX}_{STORE_NAME}"
TABLE_NAME = f"{MLPAB_DATABRICKS_SCHEMA}.{STORE_NAME}"
WAREHOUSE_ID = "8a93fc195da2ceb1"  # mlpab-grader warehouse

def create_table_and_load_data(client):
    """Create table and load item embeddings"""
    print(f"Creating table {TABLE_NAME}...")
    
    # Read items
    items = []
    with open("data/items.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append({
                "item_id": row["item_id"],
                "embedding": json.loads(row["embedding"]),
                "label": row["label"]
            })
    
    print(f"Loaded {len(items)} items")
    
    # Create the table using SQL
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        item_id STRING,
        embedding ARRAY<FLOAT>,
        label STRING
    ) USING DELTA
    """
    
    print(f"Creating table...")
    try:
        client.statement_execution.execute_statement(
            statement=create_table_sql,
            warehouse_id=WAREHOUSE_ID
        )
        print("Table created successfully")
    except Exception as e:
        print(f"Error creating table: {e}")
        # Try to drop and recreate
        client.statement_execution.execute_statement(
            statement=f"DROP TABLE IF EXISTS {TABLE_NAME}",
            warehouse_id=WAREHOUSE_ID
        )
        client.statement_execution.execute_statement(
            statement=create_table_sql,
            warehouse_id=WAREHOUSE_ID
        )
        print("Table created after drop")
    
    # Insert data using ARRAY() function
    # Batch insert to avoid SQL length limits
    batch_size = 50
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        values = []
        for item in batch:
            embedding_vals = ", ".join(str(x) for x in item["embedding"])
            values.append(f"('{item['item_id']}', ARRAY({embedding_vals}), '{item['label']}')")
        
        insert_sql = f"""
        INSERT INTO {TABLE_NAME} (item_id, embedding, label)
        VALUES {', '.join(values)}
        """
        print(f"Inserting batch {i//batch_size + 1}/{(len(items)+batch_size-1)//batch_size}")
        result = client.statement_execution.execute_statement(
            statement=insert_sql,
            warehouse_id=WAREHOUSE_ID
        )
        print(f"  Inserted {result.result.data_array[0][1]} rows")
    
    print(f"Inserted {len(items)} items into {TABLE_NAME}")
    
    # Verify
    result = client.statement_execution.execute_statement(
        statement=f"SELECT COUNT(*) as count FROM {TABLE_NAME}",
        warehouse_id=WAREHOUSE_ID
    )
    count = result.result.data_array[0][0]
    print(f"Table has {count} rows")
    
    return TABLE_NAME


def create_vector_search_index(client, table_name):
    """Create vector search index on the table"""
    print(f"Creating vector search index {INDEX_NAME} on endpoint {ENDPOINT_NAME}...")
    
    # Check if endpoint exists
    try:
        endpoint = client.vector_search_endpoints.get_endpoint(ENDPOINT_NAME)
        print(f"Using existing endpoint: {endpoint.name} (status: {endpoint.endpoint_status.state})")
    except Exception as e:
        print(f"Endpoint does not exist, creating: {e}")
        # Create endpoint
        endpoint = client.vector_search_endpoints.create_endpoint_and_wait(
            name=ENDPOINT_NAME,
            endpoint_type=EndpointType.STANDARD
        )
        print(f"Created endpoint: {endpoint.name}")
    
    # Create index spec
    index_spec = DeltaSyncVectorIndexSpecRequest(
        source_table=table_name,
        columns_to_index=["item_id", "label"],
        columns_to_sync=["item_id", "label"],
        embedding_vector_columns=[
            EmbeddingVectorColumn(
                name="embedding",
                embedding_dimension=16
            )
        ],
        pipeline_type=PipelineType.TRIGGERED
    )
    
    # Create the index
    try:
        index = client.vector_search_indexes.create_index(
            name=INDEX_NAME,
            endpoint_name=ENDPOINT_NAME,
            primary_key="item_id",
            index_type=VectorIndexType.DELTA_SYNC,
            index_subtype=IndexSubtype.HYBRID,
            delta_sync_index_spec=index_spec
        )
        print(f"Created index: {index.name}")
        
        # Wait for index to be ready
        print("Waiting for index to be ready...")
        max_retries = 60
        for i in range(max_retries):
            try:
                index_info = client.vector_search_indexes.get_index(
                    endpoint_name=ENDPOINT_NAME,
                    index_name=INDEX_NAME
                )
                print(f"Index status: {index_info.status}")
                if index_info.status == "ONLINE":
                    print("Index is online!")
                    break
            except Exception as e:
                print(f"Error checking index status: {e}")
            time.sleep(10)
        
        # Sync the index to populate it with data
        print("Syncing index...")
        sync_result = client.vector_search_indexes.sync_index(
            endpoint_name=ENDPOINT_NAME,
            index_name=INDEX_NAME
        )
        print(f"Sync result: {sync_result}")
        
        # Wait for sync to complete
        time.sleep(30)
        
        return INDEX_NAME
    except Exception as e:
        print(f"Error creating index: {e}")
        raise


def query_index(client, index_name, endpoint_name):
    """Query the index for all queries and return results"""
    print(f"Querying index {index_name}...")
    
    # Read queries
    queries = []
    with open("data/queries.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            queries.append({
                "query_id": row["query_id"],
                "embedding": json.loads(row["embedding"])
            })
    
    print(f"Loaded {len(queries)} queries")
    
    results = {}
    
    for query in queries:
        query_id = query["query_id"]
        embedding = query["embedding"]
        
        print(f"Querying for {query_id}...")
        
        try:
            response = client.vector_search_indexes.query_index(
                endpoint_name=endpoint_name,
                index_name=index_name,
                query_vector=embedding,
                filters=None,
                limit=5,
                return_distance=True
            )
            
            # Extract item_ids from results
            neighbors = []
            if hasattr(response.result, 'data_array'):
                for result in response.result.data_array:
                    print(f"  Raw result: {result}")
                    if isinstance(result, dict):
                        # Try different possible field names
                        item_id = result.get('item_id') or result.get('id') or result.get('metadata', {}).get('item_id')
                        if item_id:
                            neighbors.append(item_id)
                    elif hasattr(result, 'item_id'):
                        neighbors.append(result.item_id)
                    elif isinstance(result, (list, tuple)) and len(result) > 0:
                        neighbors.append(str(result[0]))
            
            results[query_id] = neighbors[:5]
            print(f"  Found neighbors: {neighbors[:5]}")
            
        except Exception as e:
            print(f"Error querying for {query_id}: {e}")
            import traceback
            traceback.print_exc()
            results[query_id] = []
    
    return results


def main():
    print("Starting vector search task...")
    print(f"Schema: {MLPAB_DATABRICKS_SCHEMA}")
    print(f"Prefix: {MLPAB_DATABRICKS_PREFIX}")
    print(f"Store name: {STORE_NAME}")
    print(f"Index name: {INDEX_NAME}")
    print(f"Endpoint name: {ENDPOINT_NAME}")
    print(f"Table name: {TABLE_NAME}")
    
    client = databricks.sdk.WorkspaceClient()
    
    # Step 1: Create table and load data
    table_name = create_table_and_load_data(client)
    
    # Step 2: Create vector search index
    index_name = create_vector_search_index(client, table_name)
    
    # Step 3: Query the index
    results = query_index(client, index_name, ENDPOINT_NAME)
    
    # Step 4: Write results
    output = {
        "store": STORE_NAME,
        "neighbors": results
    }
    
    os.makedirs("submission", exist_ok=True)
    with open("submission/answers.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"Results written to submission/answers.json")
    print(f"Final output: {json.dumps(output, indent=2)}")


if __name__ == "__main__":
    main()
