import databricks.sdk
import json
import csv

client = databricks.sdk.WorkspaceClient()
WAREHOUSE_ID = "8a93fc195da2ceb1"
TABLE_NAME = "workspace.mlpabd7bcb5.itemsffc8a7"

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

# Insert data using ARRAY() function
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

# Verify
result = client.statement_execution.execute_statement(
    statement=f"SELECT COUNT(*) as count FROM {TABLE_NAME}",
    warehouse_id=WAREHOUSE_ID
)
count = result.result.data_array[0][0]
print(f"Table has {count} rows")
