import databricks.sdk
import json
import csv
import os

# Environment variables
MLPAB_DATABRICKS_SCHEMA = os.environ.get("MLPAB_DATABRICKS_SCHEMA", "workspace.mlpabd7bcb5")
MLPAB_DATABRICKS_PREFIX = os.environ.get("MLPAB_DATABRICKS_PREFIX", "mlpabd7bcb5")

# Store/index name
STORE_NAME = "itemsffc8a7"
INDEX_NAME = f"{MLPAB_DATABRICKS_SCHEMA}.{STORE_NAME}_vec_idx"

print("Querying index...")

client = databricks.sdk.WorkspaceClient()

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
                print(f"  Raw result type: {type(result)}, value: {result}")
                if isinstance(result, dict):
                    item_id = result.get('item_id') or result.get('id')
                    if item_id:
                        neighbors.append(item_id)
                elif hasattr(result, 'item_id'):
                    neighbors.append(result.item_id)
                elif isinstance(result, (list, tuple)) and len(result) > 0:
                    neighbors.append(str(result[0]) if len(result) > 0 else "unknown")
        
        results[query_id] = neighbors[:5]
        print(f"  Found neighbors: {neighbors[:5]}")
        
    except Exception as e:
        print(f"Error querying for {query_id}: {e}")
        import traceback
        traceback.print_exc()
        results[query_id] = []

# Write results
output = {
    "store": STORE_NAME,
    "neighbors": results
}

os.makedirs("submission", exist_ok=True)
with open("submission/answers.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"Results written to submission/answers.json")
print(f"Final output:")
print(json.dumps(output, indent=2))
