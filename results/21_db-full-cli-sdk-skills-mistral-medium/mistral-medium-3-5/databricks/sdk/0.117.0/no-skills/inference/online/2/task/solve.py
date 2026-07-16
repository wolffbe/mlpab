#!/usr/bin/env python3
import os
import json
import time
import databricks.sdk
from databricks.sdk.service.catalog import OnlineTable, OnlineTableSpec
from databricks.sdk.service.sql import Disposition, Format

# Initialize workspace client
w = databricks.sdk.WorkspaceClient()

# Environment variables
SCHEMA = os.environ['MLPAB_DATABRICKS_SCHEMA']
PREFIX = os.environ['MLPAB_DATABRICKS_PREFIX']

# Parse schema
catalog, schema = SCHEMA.split('.')

# Table names
SOURCE_TABLE_NAME = 'profilesd1eca7_source'
FEATURE_TABLE_NAME = 'profilesd1eca7'

print(f"Schema: {SCHEMA}")
print(f"Catalog: {catalog}, Schema: {schema}")

# Helper function to wait for statement completion
def wait_for_statement(statement_id, timeout=600):
    start = time.time()
    while time.time() - start < timeout:
        status_resp = w.statement_execution.get_statement(statement_id=statement_id)
        status = status_resp.status.state.value
        if status in ['SUCCESS', 'FAILED', 'CANCELED']:
            if status == 'FAILED':
                error_msg = status_resp.status.error_message or "Unknown error"
                raise Exception(f"Statement failed: {error_msg}")
            return status_resp
        time.sleep(10)
    raise Exception("Statement execution timed out")

# Helper function to execute SQL and get results as JSON array
def execute_sql_and_get_results(sql, warehouse_id='8a93fc195da2ceb1', timeout=300):
    print(f"  Executing: {sql[:100]}...")
    result = w.statement_execution.execute_statement(
        statement=sql,
        warehouse_id=warehouse_id,
        disposition=Disposition.INLINE,
        format=Format.JSON_ARRAY,
        wait_timeout='0s'
    )
    
    wait_for_statement(result.statement_id, timeout)
    
    result_data = w.statement_execution.get_statement_result_chunk_n(
        statement_id=result.statement_id,
        chunk_index=0
    )
    
    return result_data

# Step 1: Create source table directly from CSV data
print("\n=== Step 1: Creating source table from CSV data ===")
source_table_full_name = f"{catalog}.{schema}.{SOURCE_TABLE_NAME}"

# Read CSV data
import csv
with open('data/features.csv', 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Build a single CREATE TABLE AS SELECT statement with UNION ALL
# Split into multiple statements to avoid hitting SQL length limits
batch_size = 30
for batch_num in range(0, len(rows), batch_size):
    batch = rows[batch_num:batch_num+batch_size]
    
    # Create a temp table for this batch
    temp_table_name = f"{catalog}.{schema}.{SOURCE_TABLE_NAME}_batch_{batch_num//batch_size}"
    
    select_clauses = []
    for row in batch:
        select_clauses.append(f"SELECT '{row['account_id']}' AS account_id, {row['f1']} AS f1, {row['f2']} AS f2, {row['f3']} AS f3, {row['f4']} AS f4")
    
    create_batch_sql = f"CREATE TABLE {temp_table_name} AS {" UNION ALL ".join(select_clauses)}"
    
    print(f"  Creating batch table {batch_num//batch_size + 1}")
    result = w.statement_execution.execute_statement(
        statement=create_batch_sql,
        warehouse_id='8a93fc195da2ceb1',
        wait_timeout='0s'
    )
    wait_for_statement(result.statement_id)

# Now combine all batch tables into the final source table
# First, get all batch table names
batch_table_names = [f"{catalog}.{schema}.{SOURCE_TABLE_NAME}_batch_{i}" for i in range(0, len(rows), batch_size)]

# Create the final table by selecting from all batch tables
final_create_sql = f"""
CREATE TABLE IF NOT EXISTS {source_table_full_name} AS
SELECT * FROM {batch_table_names[0]}
"""

for table in batch_table_names[1:]:
    final_create_sql += f" UNION ALL SELECT * FROM {table}\n"

print(f"  Creating final source table")
result = w.statement_execution.execute_statement(
    statement=final_create_sql,
    warehouse_id='8a93fc195da2ceb1',
    wait_timeout='0s'
)
wait_for_statement(result.statement_id)
print(f"Source table created: {source_table_full_name}")

# Step 3: Create online table
print("\n=== Step 3: Creating online table ===")
online_table_full_name = f"{catalog}.{schema}.{FEATURE_TABLE_NAME}"

online_table_spec = OnlineTableSpec(
    source_table_full_name=source_table_full_name,
    primary_key_columns=['account_id'],
    perform_full_copy=True
)

online_table = OnlineTable(
    name=online_table_full_name,
    spec=online_table_spec
)

print(f"Creating online table: {online_table_full_name}")
create_op = w.online_tables.create(table=online_table)

# Wait for online table to be active
print("Waiting for online table to be active...")
def wait_for_online_table(table_name, timeout=600):
    start = time.time()
    while time.time() - start < timeout:
        try:
            table_info = w.online_tables.get(name=table_name)
            if table_info.status and table_info.status.state:
                state = table_info.status.state.value
                print(f"  Online table state: {state}")
                if state == 'ACTIVE':
                    return table_info
            time.sleep(5)
        except Exception as e:
            print(f"  Error checking table status: {e}")
            time.sleep(5)
    raise Exception("Online table creation timed out")

online_table_info = wait_for_online_table(online_table_full_name)
print(f"Online table created and active: {online_table_full_name}")

# Step 4: Read lookup keys
print("\n=== Step 4: Reading lookup keys ===")
with open('data/lookup_keys.txt', 'r') as f:
    lookup_keys = [line.strip() for line in f.readlines()]

print(f"Lookup keys: {lookup_keys}")

# Step 5: Query online table for each key
print("\n=== Step 5: Querying online table ===")
vectors = {}

for key in lookup_keys:
    query = f"""
    SELECT f1, f2, f3, f4 
    FROM {online_table_full_name} 
    WHERE account_id = '{key}'
    """
    
    print(f"Querying for key: {key}")
    result_data = execute_sql_and_get_results(query)
    
    if result_data.data_array and len(result_data.data_array) > 0:
        row = result_data.data_array[0]
        if len(row) == 4:
            vector = [float(row[0]), float(row[1]), float(row[2]), float(row[3])]
            vectors[key] = vector
            print(f"  Found vector for {key}: {vector}")
        else:
            print(f"  Unexpected row length for {key}: {len(row)}")
            vectors[key] = [None, None, None, None]
    else:
        print(f"  No data found for {key}")
        vectors[key] = [None, None, None, None]

# Step 6: Write results to submission/answers.json
print("\n=== Step 6: Writing results ===")
os.makedirs('submission', exist_ok=True)
output = {"vectors": vectors}

with open('submission/answers.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"Results written to submission/answers.json")
print(f"Vectors: {json.dumps(output, indent=2)}")
print("\nDone!")
