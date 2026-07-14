#!/usr/bin/env python3

import csv
import json
import os
from databricks.sdk import WorkspaceClient

def filter_data():
    """Filter the CSV to only include valid rows and return the filtered data."""
    valid_categories = {'grocery', 'travel', 'salary', 'rent', 'other'}
    valid_rows = []
    rejected_rows = []
    
    with open('data/events.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_id = row['row_id']
            amount_str = row['amount']
            category = row['category']
            
            # Rule 1: amount is present (not null/empty)
            if amount_str.strip() == '':
                rejected_rows.append(row_id)
                continue
            
            # Rule 2: amount is within [0, 10000] (inclusive)
            try:
                amount = float(amount_str)
                if amount < 0 or amount > 10000:
                    rejected_rows.append(row_id)
                    continue
            except ValueError:
                rejected_rows.append(row_id)
                continue
            
            # Rule 3: category is one of the valid ones
            if category not in valid_categories:
                rejected_rows.append(row_id)
                continue
            
            # If we get here, the row is valid
            valid_rows.append(row)
    
    return valid_rows, rejected_rows

def create_answers_json(rejected_rows):
    """Create the answers.json file with rejected row IDs."""
    # Sort the rejected rows
    rejected_rows.sort()
    
    # Create submission directory if it doesn't exist
    os.makedirs('submission', exist_ok=True)
    
    # Write the answers.json file
    with open('submission/answers.json', 'w') as f:
        json.dump({"rejected": rejected_rows}, f)
    
    print(f"✓ Created submission/answers.json with {len(rejected_rows)} rejected rows")

def main():
    print("Starting clean feature table creation process...")
    
    # Step 1: Filter the data and create answers.json
    valid_rows, rejected_rows = filter_data()
    create_answers_json(rejected_rows)
    
    print(f"✓ Valid rows: {len(valid_rows)}")
    print(f"✓ Rejected rows: {len(rejected_rows)}")
    
    # Step 2: Connect to Databricks
    ws = WorkspaceClient()
    print("✓ Connected to Databricks workspace")
    
    # Get environment variables
    schema_name = os.environ.get('MLPAB_DATABRICKS_SCHEMA', 'workspace.mlpab774429')
    prefix = os.environ.get('MLPAB_DATABRICKS_PREFIX', 'mlpab774429')
    
    # Parse schema to get catalog and schema parts
    if '.' in schema_name:
        catalog_name, schema_part = schema_name.split('.', 1)
    else:
        catalog_name = None
        schema_part = schema_name
    
    print(f"✓ Catalog: {catalog_name}")
    print(f"✓ Schema: {schema_part}")
    
    # Get warehouse
    warehouses = list(ws.warehouses.list())
    warehouse_id = warehouses[0].id if warehouses else None
    print(f"✓ Warehouse ID: {warehouse_id}")
    
    # Step 3: Drop existing table if it exists
    table_full_name = f"{catalog_name}.{schema_part}.eventsa45e2a"
    
    drop_sql = f"DROP TABLE IF EXISTS {table_full_name}"
    
    try:
        result = ws.statement_execution.execute_statement(
            warehouse_id=warehouse_id,
            catalog=catalog_name,
            schema=schema_part,
            statement=drop_sql,
            wait_timeout="30s"
        )
        print(f"✓ Dropped existing table if it existed")
    except Exception as e:
        print(f"✗ Error dropping table: {e}")
    
    # Step 4: Create the feature table using SQL
    create_table_sql = f"""
CREATE TABLE IF NOT EXISTS {table_full_name} (
    row_id STRING,
    account_id STRING,
    event_time BIGINT,
    amount DOUBLE,
    category STRING
) USING DELTA
COMMENT 'Feature table for valid events data'
"""
    
    print(f"✓ Creating empty table...")
    
    try:
        result = ws.statement_execution.execute_statement(
            warehouse_id=warehouse_id,
            catalog=catalog_name,
            schema=schema_part,
            statement=create_table_sql,
            wait_timeout="30s"
        )
        print(f"✓ Empty table created successfully")
    except Exception as e:
        print(f"✗ Error creating empty table: {e}")
        return
    
    # Step 5: Insert data in batches
    batch_size = 10  # Small batches to avoid issues
    total_batches = (len(valid_rows) + batch_size - 1) // batch_size
    
    for i in range(0, len(valid_rows), batch_size):
        batch = valid_rows[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        print(f"✓ Processing batch {batch_num}/{total_batches}...")
        
        # Generate VALUES clause
        values_clauses = []
        for row in batch:
            row_id = row['row_id'].replace("'", "''")
            account_id = row['account_id'].replace("'", "''")
            event_time = row['event_time']
            amount = row['amount']
            category = row['category'].replace("'", "''")
            
            values_clauses.append(f"('{row_id}', '{account_id}', {event_time}, {amount}, '{category}')")
        
        values_sql = ", ".join(values_clauses)
        
        insert_sql = f"""
INSERT INTO {table_full_name} (row_id, account_id, event_time, amount, category)
VALUES {values_sql}
"""
        
        try:
            result = ws.statement_execution.execute_statement(
                warehouse_id=warehouse_id,
                catalog=catalog_name,
                schema=schema_part,
                statement=insert_sql,
                wait_timeout="30s"
            )
            print(f"✓ Batch {batch_num} inserted successfully")
        except Exception as e:
            print(f"✗ Error inserting batch {batch_num}: {e}")
            # Try each row individually
            for j, single_row in enumerate(batch):
                row_id = single_row['row_id'].replace("'", "''")
                account_id = single_row['account_id'].replace("'", "''")
                event_time = single_row['event_time']
                amount = single_row['amount']
                category = single_row['category'].replace("'", "''")
                
                single_insert_sql = f"""
INSERT INTO {table_full_name} (row_id, account_id, event_time, amount, category)
VALUES ('{row_id}', '{account_id}', {event_time}, {amount}, '{category}')
"""
                
                try:
                    result = ws.statement_execution.execute_statement(
                        warehouse_id=warehouse_id,
                        catalog=catalog_name,
                        schema=schema_part,
                        statement=single_insert_sql,
                        wait_timeout="30s"
                    )
                    print(f"✓ Row {row_id} inserted successfully")
                except Exception as e2:
                    print(f"✗ Error inserting row {row_id}: {e2}")
                    break  # Stop if we keep getting errors
    
    # Step 6: Verify the table has the correct number of rows
    verify_sql = f"SELECT COUNT(*) as count FROM {table_full_name}"
    
    try:
        result = ws.statement_execution.execute_statement(
            warehouse_id=warehouse_id,
            catalog=catalog_name,
            schema=schema_part,
            statement=verify_sql,
            wait_timeout="30s"
        )
        
        if hasattr(result, 'result') and hasattr(result.result, 'data_array') and result.result.data_array:
            row_count = result.result.data_array[0][0]
            print(f"✓ Table has {row_count} rows (expected: {len(valid_rows)})")
            
            if int(row_count) == len(valid_rows):
                print("✓ Row count matches expected valid rows!")
            else:
                print(f"✗ Row count mismatch! Expected {len(valid_rows)}, got {row_count}")
        else:
            print("✗ Could not verify row count")
            
    except Exception as e:
        print(f"✗ Error verifying table: {e}")
    
    print("✓ Script completed successfully")

if __name__ == "__main__":
    main()
