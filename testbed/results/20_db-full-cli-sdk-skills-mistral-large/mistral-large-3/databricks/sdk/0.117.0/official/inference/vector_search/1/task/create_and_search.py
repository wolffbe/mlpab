#!/usr/bin/env python3
"""
Load item embeddings into a Databricks Delta table, create a vector search endpoint and index,
and perform similarity searches for all queries.
"""

import databricks.sdk
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import vectorsearch, catalog
import json
import os
import time
import pandas as pd
import io

# Environment variables
SCHEMA = os.getenv("MLPAB_DATABRICKS_SCHEMA")  # workspace.mlpab8dd220
PREFIX = os.getenv("MLPAB_DATABRICKS_PREFIX")  # mlpab8dd220
CATALOG, SCHEMA_NAME = SCHEMA.split(".")

# Resource names
TABLE_NAME = f"items568016_table"
FULL_TABLE_NAME = f"{CATALOG}.{SCHEMA_NAME}.{TABLE_NAME}"
VOLUME_NAME = f"items568016_volume"
FULL_VOLUME_NAME = f"{CATALOG}.{SCHEMA_NAME}.{VOLUME_NAME}"
ENDPOINT_NAME = f"{PREFIX}_items568016_endpoint"
INDEX_NAME = f"{PREFIX}_items568016_index"

# Initialize WorkspaceClient
w = WorkspaceClient()

# Step 1: Create Volume and Delta table from items.csv
def create_delta_table():
    df = pd.read_csv("data/items.csv")
    
    # Save the CSV locally
    temp_file = f"items.csv"
    df.to_csv(temp_file, index=False)
    
    # Create a volume in the schema
    volumes = w.volumes.list(catalog_name=CATALOG, schema_name=SCHEMA_NAME)
    if not any(vol.name == VOLUME_NAME for vol in volumes):
        print(f"Creating volume: {FULL_VOLUME_NAME}")
        w.volumes.create(
            catalog_name=CATALOG,
            schema_name=SCHEMA_NAME,
            name=VOLUME_NAME,
            volume_type=catalog.VolumeType.MANAGED
        )
    else:
        print(f"Volume already exists: {FULL_VOLUME_NAME}")
    
    # Upload the file to the volume using the Files API
    with open(temp_file, "rb") as f:
        file_bytes = f.read()
    file_io = io.BytesIO(file_bytes)
    w.files.upload(
        f"/Volumes/{CATALOG}/{SCHEMA_NAME}/{VOLUME_NAME}/{PREFIX}_items.csv", 
        file_io, 
        overwrite=True
    )
    
    # Create the table using SQL
    warehouses = list(w.warehouses.list())
    if not warehouses:
        raise RuntimeError("No warehouses available")
    
    spark_sql = w.statement_execution.execute_statement(
        warehouse_id=warehouses[0].id,
        catalog=CATALOG,
        schema=SCHEMA_NAME,
        statement=f"""
        CREATE TABLE IF NOT EXISTS {FULL_TABLE_NAME} (
            item_id STRING,
            embedding STRING,
            label STRING
        )
        USING DELTA
        """
    )
    
    # Wait for the statement to complete
    while True:
        status = w.statement_execution.get_statement(spark_sql.statement_id)
        if status.status.state in ["SUCCEEDED", "FAILED", "CANCELED"]:
            break
        time.sleep(5)
    
    # Load data into the table using SQL
    spark_sql = w.statement_execution.execute_statement(
        warehouse_id=warehouses[0].id,
        catalog=CATALOG,
        schema=SCHEMA_NAME,
        statement=f"""
        COPY INTO {FULL_TABLE_NAME}
        FROM '/Volumes/{CATALOG}/{SCHEMA_NAME}/{VOLUME_NAME}/{PREFIX}_items.csv'
        FILEFORMAT = CSV
        FORMAT_OPTIONS ('header' = 'true', 'inferSchema' = 'false')
        """
    )
    
    # Wait for the statement to complete
    while True:
        status = w.statement_execution.get_statement(spark_sql.statement_id)
        if status.status.state in ["SUCCEEDED", "FAILED", "CANCELED"]:
            break
        time.sleep(5)
    
    print(f"Created and loaded data into table: {FULL_TABLE_NAME}")

# Step 2: Create vector search endpoint
def create_vector_search_endpoint():
    vs_client = w.vector_search_endpoints
    
    # Check if endpoint already exists
    endpoints = vs_client.list_endpoints().endpoints
    if not any(ep.name == ENDPOINT_NAME for ep in endpoints):
        print(f"Creating vector search endpoint: {ENDPOINT_NAME}")
        vs_client.create_endpoint(
            name=ENDPOINT_NAME,
            endpoint_type=vectorsearch.EndpointType.STANDARD
        )
        
        # Wait for endpoint to be ready
        while True:
            ep = vs_client.get_endpoint(ENDPOINT_NAME)
            if ep.endpoint_status.state == vectorsearch.EndpointStatusState.ONLINE:
                break
            print(f"Endpoint status: {ep.endpoint_status.state}")
            time.sleep(10)
    else:
        print(f"Vector search endpoint already exists: {ENDPOINT_NAME}")

# Step 3: Create vector search index
def create_vector_search_index():
    vs_client = w.vector_search_indexes
    
    # Check if index already exists
    indexes = vs_client.list_indexes(endpoint_name=ENDPOINT_NAME).vector_indexes
    if not any(idx.name == INDEX_NAME for idx in indexes):
        print(f"Creating vector search index: {INDEX_NAME}")
        vs_client.create_index(
            endpoint_name=ENDPOINT_NAME,
            name=INDEX_NAME,
            primary_key="item_id",
            index_type=vectorsearch.IndexType.DELTA_SYNC,
            delta_sync_index_spec=vectorsearch.DeltaSyncIndexSpec(
                source_table=FULL_TABLE_NAME,
                embedding_source_column="embedding",
                embedding_vector_column="embedding",
                embedding_dimension=16
            )
        )
        
        # Wait for index to be ready
        while True:
            idx = vs_client.get_index(ENDPOINT_NAME, INDEX_NAME)
            if idx.status.ready:
                break
            print(f"Index status: {idx.status}")
            time.sleep(10)
    else:
        print(f"Vector search index already exists: {INDEX_NAME}")

# Step 4: Perform similarity searches for all queries
def perform_similarity_searches():
    queries_df = pd.read_csv("data/queries.csv")
    queries_df['embedding'] = queries_df['embedding'].apply(lambda x: json.loads(x))
    
    vs_client = w.vector_search_indexes
    index = vs_client.get_index(ENDPOINT_NAME, INDEX_NAME)
    
    neighbors = {}
    for _, row in queries_df.iterrows():
        query_id = row['query_id']
        embedding = row['embedding']
        
        results = index.similarity_search(
            query_vector=embedding,
            columns=["item_id"],
            num_results=5
        )
        
        # Extract item_ids
        item_ids = [result['item_id'] for result in results.result.data_array]
        neighbors[query_id] = item_ids
    
    # Write results to submission/answers.json
    output = {
        "store": INDEX_NAME,
        "neighbors": neighbors
    }
    
    with open("submission/answers.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print("Wrote results to submission/answers.json")

if __name__ == "__main__":
    create_delta_table()
    create_vector_search_endpoint()
    create_vector_search_index()
    perform_similarity_searches()