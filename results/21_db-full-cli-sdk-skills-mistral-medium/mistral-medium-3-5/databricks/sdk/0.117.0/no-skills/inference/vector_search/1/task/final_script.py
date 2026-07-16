import databricks.sdk
from databricks.sdk.service.vectorsearch import (
    VectorIndexType,
    IndexSubtype,
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingVectorColumn,
    PipelineType,
    EndpointType,
)
import json
import csv
import time
import os

# Environment variables
MLPAB_DATABRICKS_SCHEMA = os.environ.get("MLPAB_DATABRICKS_SCHEMA", "workspace.mlpabd7bcb5")
MLPAB_DATABRICKS_PREFIX = os.environ.get("MLPAB_DATABRICKS_PREFIX", "mlpabd7bcb5")

# Store/index name
STORE_NAME = "itemsffc8a7"
INDEX_NAME = f"{MLPAB_DATABRICKS_SCHEMA}.{STORE_NAME}_vec_idx"  # Unique index name
ENDPOINT_NAME = f"{MLPAB_DATABRICKS_PREFIX}_{STORE_NAME}"
TABLE_NAME = f"{MLPAB_DATABRICKS_SCHEMA}.{STORE_NAME}"
WAREHOUSE_ID = "8a93fc195da2ceb1"  # mlpab-grader warehouse

print("Starting final script...")
print(f"Schema: {MLPAB_DATABRICKS_SCHEMA}")
print(f"Prefix: {MLPAB_DATABRICKS_PREFIX}")
print(f"Store name: {STORE_NAME}")
print(f"Index name: {INDEX_NAME}")
print(f"Endpoint name: {ENDPOINT_NAME}")
print(f"Table name: {TABLE_NAME}")

client = databricks.sdk.WorkspaceClient()

# Step 1: Ensure table exists with CDF enabled
print("\n=== Step 1: Creating table ===")
try:
    client.statement_execution.execute_statement(
        statement=f"DROP TABLE IF EXISTS {TABLE_NAME}",
        warehouse_id=WAREHOUSE_ID
    )
    print("Dropped old table")
except Exception as e:
    print(f"Error dropping table: {e}")

client.statement_execution.execute_statement(
    statement=f"""
    CREATE TABLE {TABLE_NAME} (
        item_id STRING,
        embedding ARRAY<FLOAT>,
        label STRING
    ) USING DELTA
    TBLPROPERTIES (delta.enableChangeDataFeed = true)
    """,
    warehouse_id=WAREHOUSE_ID
)
print("Table created with CDF enabled")

# Step 2: Load data
print("\n=== Step 2: Loading data ===")
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
    result = client.statement_execution.execute_statement(
        statement=insert_sql,
        warehouse_id=WAREHOUSE_ID
    )

result = client.statement_execution.execute_statement(
    statement=f"SELECT COUNT(*) as count FROM {TABLE_NAME}",
    warehouse_id=WAREHOUSE_ID
)
count = result.result.data_array[0][0]
print(f"Table has {count} rows")

# Step 3: Ensure endpoint exists
print("\n=== Step 3: Creating endpoint ===")
try:
    endpoint = client.vector_search_endpoints.get_endpoint(ENDPOINT_NAME)
    print(f"Using existing endpoint: {endpoint.name} (status: {endpoint.endpoint_status.state})")
except Exception as e:
    print(f"Creating new endpoint: {e}")
    endpoint = client.vector_search_endpoints.create_endpoint_and_wait(
        name=ENDPOINT_NAME,
        endpoint_type=EndpointType.STANDARD
    )
    print(f"Created endpoint: {endpoint.name}")

# Step 4: Create index
print("\n=== Step 4: Creating index ===")

# Delete old index if it exists
try:
    client.vector_search_indexes.delete_index(INDEX_NAME)
    print(f"Deleted old index {INDEX_NAME}")
except Exception as e:
    print(f"No old index to delete: {e}")

index_spec = DeltaSyncVectorIndexSpecRequest(
    source_table=TABLE_NAME,
    columns_to_index=["item_id", "label"],
    embedding_vector_columns=[
        EmbeddingVectorColumn(
            name="embedding",
            embedding_dimension=16
        )
    ],
    pipeline_type=PipelineType.TRIGGERED
)

index = client.vector_search_indexes.create_index(
    name=INDEX_NAME,
    endpoint_name=ENDPOINT_NAME,
    primary_key="item_id",
    index_type=VectorIndexType.DELTA_SYNC,
    index_subtype=IndexSubtype.HYBRID,
    delta_sync_index_spec=index_spec
)
print(f"Created index: {index.name}")

# Step 5: Wait for index to be ready
print("\n=== Step 5: Waiting for index to be ready ===")
max_retries = 120  # 20 minutes
for i in range(max_retries):
    try:
        index_info = client.vector_search_indexes.get_index(index_name=INDEX_NAME)
        print(f"Index status: {index_info.status}")
        if index_info.status.ready:
            print("Index is ready!")
            break
    except Exception as e:
        print(f"Error checking index status: {e}")
    time.sleep(10)

# Step 6: Sync the index
print("\n=== Step 6: Syncing index ===")
try:
    sync_result = client.vector_search_indexes.sync_index(index_name=INDEX_NAME)
    print(f"Sync result: {sync_result}")
except Exception as e:
    print(f"Error syncing: {e}")

# Wait for sync to complete
print("Waiting for sync to complete...")
for i in range(60):
    try:
        index_info = client.vector_search_indexes.get_index(index_name=INDEX_NAME)
        if index_info.status.indexed_row_count is not None and int(index_info.status.indexed_row_count) > 0:
            print(f"Index has {index_info.status.indexed_row_count} rows indexed!")
            break
    except Exception as e:
        pass
    time.sleep(10)

# Step 7: Query the index
print("\n=== Step 7: Querying index ===")

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
            index_name=INDEX_NAME,
            query_vector=embedding,
            columns=["item_id"],
            num_results=5
        )
        
        # Extract item_ids from results
        neighbors = []
        if hasattr(response.result, 'data_array'):
            for result in response.result.data_array:
                # The result structure might vary, try to extract item_id
                if isinstance(result, dict):
                    item_id = result.get('item_id') or result.get('id')
                    if item_id:
                        neighbors.append(item_id)
                elif hasattr(result, 'item_id'):
                    neighbors.append(result.item_id)
                elif isinstance(result, (list, tuple)) and len(result) > 0:
                    # Try to get item_id from the result
                    neighbors.append(str(result[0]) if len(result) > 0 else "unknown")
        
        results[query_id] = neighbors[:5]
        print(f"  Found neighbors: {neighbors[:5]}")
        
    except Exception as e:
        print(f"Error querying for {query_id}: {e}")
        import traceback
        traceback.print_exc()
        results[query_id] = []

# Step 8: Write results
print("\n=== Step 8: Writing results ===")
output = {
    "store": STORE_NAME,
    "neighbors": results
}

os.makedirs("submission", exist_ok=True)
with open("submission/answers.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"Results written to submission/answers.json")
print(f"Final output preview:")
print(json.dumps(output, indent=2)[:500])
