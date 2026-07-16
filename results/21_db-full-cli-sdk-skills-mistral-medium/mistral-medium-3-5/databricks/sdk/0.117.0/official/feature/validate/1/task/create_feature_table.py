#!/usr/bin/env python3

import csv
import json
import os
import tempfile
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import TableType, DataSourceFormat, OnlineTable, OnlineTableSpec

def filter_and_create_valid_csv():
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
        json.dump({"rejected": rejected_rows}, f, indent=2)
    
    print(f"Created submission/answers.json with {len(rejected_rows)} rejected rows")

def main():
    print("Starting feature table creation process...")
    
    # Step 1: Filter the data and create answers.json
    valid_rows, rejected_rows = filter_and_create_valid_csv()
    create_answers_json(rejected_rows)
    
    print(f"Valid rows: {len(valid_rows)}")
    print(f"Rejected rows: {len(rejected_rows)}")
    
    # Step 2: Connect to Databricks
    ws = WorkspaceClient()
    print("Connected to Databricks workspace")
    
    # Get environment variables
    schema_name = os.environ.get('MLPAB_DATABRICKS_SCHEMA', 'workspace.mlpab774429')
    prefix = os.environ.get('MLPAB_DATABRICKS_PREFIX', 'mlpab774429')
    
    # Parse schema to get catalog and schema parts
    if '.' in schema_name:
        catalog_name, schema_part = schema_name.split('.', 1)
    else:
        catalog_name = None
        schema_part = schema_name
    
    print(f"Catalog: {catalog_name}")
    print(f"Schema: {schema_part}")
    
    # Step 3: Create a temporary CSV with only valid data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_csv:
        if valid_rows:
            # Write header
            fieldnames = valid_rows[0].keys()
            writer = csv.DictWriter(temp_csv, fieldnames=fieldnames)
            writer.writeheader()
            
            # Write valid rows
            for row in valid_rows:
                writer.writerow(row)
        
        temp_csv_path = temp_csv.name
    
    print(f"Created temporary CSV with valid data: {temp_csv_path}")
    
    # Step 4: Upload the CSV to DBFS
    dbfs_path = f"/FileStore/tables/{prefix}/eventsa45e2a_valid_data.csv"
    
    # Read the temp file and upload to DBFS
    with open(temp_csv_path, 'r') as f:
        content = f.read()
    
    # Upload to DBFS using workspace client
    try:
        ws.files.upload(dbfs_path, content.encode('utf-8'), overwrite=True)
        print(f"Uploaded valid data to DBFS: {dbfs_path}")
    except Exception as e:
        print(f"Error uploading to DBFS: {e}")
        # Try alternative approach - upload to workspace
        workspace_path = f"/Users/{ws.current_user().user_name}/{prefix}/eventsa45e2a_valid_data.csv"
        ws.files.upload(workspace_path, content.encode('utf-8'), overwrite=True)
        print(f"Uploaded valid data to workspace: {workspace_path}")
        dbfs_path = workspace_path
    
    # Clean up temp file
    os.unlink(temp_csv_path)
    
    # Step 5: Create the feature table using SQL
    # We'll use the workspace client to run SQL commands
    
    # First, let's check if the schema exists
    try:
        schema_info = ws.catalogs.get_schema(catalog_name=catalog_name, schema_name=schema_part)
        print(f"Schema exists: {schema_info}")
    except Exception as e:
        print(f"Schema doesn't exist or error: {e}")
        # Try to create it
        try:
            ws.catalogs.create_schema(
                catalog_name=catalog_name,
                schema_name=schema_part,
                comment=f"Schema for {prefix} run"
            )
            print(f"Created schema: {schema_name}")
        except Exception as e2:
            print(f"Error creating schema: {e2}")
    
    # Step 6: Create the table using SQL
    # We need to use the SQL execution API
    # Let's check what SQL execution methods are available
    print("Available SQL methods:")
    for attr in sorted(dir(ws)):
        if 'sql' in attr.lower() or 'query' in attr.lower() or 'execute' in attr.lower():
            print(f"  {attr}")
    
    # Try to use the warehouse or SQL execution
    try:
        # Check if we can use the SQL execution API
        if hasattr(ws, 'statement_execution'):
            print("Using statement_execution API")
            
            # Create table SQL
            table_full_name = f"{schema_name}.eventsa45e2a"
            
            # First, let's try to create the table from the uploaded CSV
            # We'll use the workspace path approach
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
            
            print(f"Creating table with SQL: {create_table_sql}")
            
            # Execute the SQL
            result = ws.statement_execution.execute_statement(
                warehouse_id="default",  # This might need to be specified differently
                catalog=catalog_name,
                schema=schema_part,
                statement=create_table_sql
            )
            print(f"Table creation result: {result}")
            
        else:
            print("statement_execution not available")
            
    except Exception as e:
        print(f"Error with SQL execution: {e}")
    
    print("Script completed")

if __name__ == "__main__":
    main()
