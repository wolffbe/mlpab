#!/usr/bin/env python3

import csv
import json
import os
import tempfile
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import OnlineTable, OnlineTableSpec
from databricks.sdk.service.sql import Disposition, Format

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
    print("Starting feature table creation process...")
    
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
    
    # Get current user
    current_user = ws.current_user.me().user_name
    print(f"✓ Current user: {current_user}")
    
    # Get warehouse
    warehouses = list(ws.warehouses.list())
    warehouse_id = warehouses[0].id if warehouses else None
    print(f"✓ Warehouse ID: {warehouse_id}")
    
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
    
    print(f"✓ Created temporary CSV with valid data: {temp_csv_path}")
    
    # Step 4: Upload the CSV to workspace storage
    workspace_path = f"/Users/{current_user}/{prefix}/eventsa45e2a_valid_data.csv"
    
    # Upload to workspace
    with open(temp_csv_path, 'rb') as f:
        ws.files.upload(workspace_path, f, overwrite=True)
    print(f"✓ Uploaded valid data to workspace: {workspace_path}")
    
    # Clean up temp file
    os.unlink(temp_csv_path)
    
    # Step 5: Check if schema exists, create if not
    try:
        schema_info = ws.catalogs.get_schema(catalog_name=catalog_name, schema_name=schema_part)
        print(f"✓ Schema exists: {schema_info.full_name}")
    except Exception as e:
        print(f"✓ Schema doesn't exist, creating: {e}")
        # Try to create it
        try:
            ws.catalogs.create_schema(
                catalog_name=catalog_name,
                schema_name=schema_part,
                comment=f"Schema for {prefix} run"
            )
            print(f"✓ Created schema: {schema_name}")
        except Exception as e2:
            print(f"✗ Error creating schema: {e2}")
            # Maybe the catalog doesn't exist, try to create it
            try:
                ws.catalogs.create_catalog(
                    name=catalog_name,
                    comment=f"Catalog for {prefix} run"
                )
                print(f"✓ Created catalog: {catalog_name}")
                
                # Now create the schema
                ws.catalogs.create_schema(
                    catalog_name=catalog_name,
                    schema_name=schema_part,
                    comment=f"Schema for {prefix} run"
                )
                print(f"✓ Created schema: {schema_name}")
            except Exception as e3:
                print(f"✗ Error creating catalog: {e3}")
    
    # Step 6: Create the feature table using SQL
    table_full_name = f"{catalog_name}.{schema_part}.eventsa45e2a"
    
    # First, create the table
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
    
    print(f"✓ Creating table with SQL...")
    
    try:
        result = ws.statement_execution.execute_statement(
            warehouse_id=warehouse_id,
            catalog=catalog_name,
            schema=schema_part,
            statement=create_table_sql,
            wait_timeout="30s"
        )
        print(f"✓ Table creation initiated: {result}")
    except Exception as e:
        print(f"✗ Error creating table: {e}")
    
    # Step 7: Load the valid data into the table
    # We'll use COPY INTO to load from the workspace file
    load_data_sql = f"""
COPY INTO {table_full_name}
FROM '{workspace_path}'
FILEFORMAT = CSV
FORMAT_OPTIONS ('header' = 'true', 'inferSchema' = 'true')
"""
    
    print(f"✓ Loading data into table...")
    
    try:
        result = ws.statement_execution.execute_statement(
            warehouse_id=warehouse_id,
            catalog=catalog_name,
            schema=schema_part,
            statement=load_data_sql,
            wait_timeout="60s"
        )
        print(f"✓ Data loading initiated: {result}")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        # Try alternative approach - create table with data directly
        create_with_data_sql = f"""
CREATE OR REPLACE TABLE {table_full_name} (
    row_id STRING,
    account_id STRING,
    event_time BIGINT,
    amount DOUBLE,
    category STRING
) USING DELTA
COMMENT 'Feature table for valid events data'
AS SELECT * FROM csv.`{workspace_path}`
"""
        
        try:
            result = ws.statement_execution.execute_statement(
                warehouse_id=warehouse_id,
                catalog=catalog_name,
                schema=schema_part,
                statement=create_with_data_sql,
                wait_timeout="60s"
            )
            print(f"✓ Table with data creation initiated: {result}")
        except Exception as e2:
            print(f"✗ Error creating table with data: {e2}")
    
    # Step 8: Create online table for low-latency access
    online_table_name = f"{catalog_name}.{schema_part}.eventsa45e2a_online"
    
    print(f"✓ Creating online table...")
    
    try:
        online_table = OnlineTable(
            name=online_table_name,
            spec=OnlineTableSpec(
                source_table_full_name=table_full_name,
                primary_key_columns=["row_id"],
                timeseries_key="event_time",
                perform_full_copy=True
            )
        )
        
        result = ws.online_tables.create(online_table)
        print(f"✓ Online table creation initiated: {result}")
        
        # Wait for the online table to be active
        wait_result = ws.online_tables.wait_get_online_table_active(online_table_name)
        print(f"✓ Online table is active: {wait_result}")
        
    except Exception as e:
        print(f"✗ Error creating online table: {e}")
        # Try alternative approach using feature store
        try:
            print("✓ Trying feature store approach...")
            
            # First, check if we need to create an online store
            online_stores = ws.feature_store.list_online_stores()
            if not online_stores:
                # Create an online store
                online_store = ws.feature_store.create_online_store(
                    name=f"{prefix}_online_store",
                    storage_type="DATABRICKS"
                )
                print(f"✓ Created online store: {online_store}")
            else:
                online_store_name = online_stores[0].name
                print(f"✓ Using existing online store: {online_store_name}")
            
            # Publish the table to feature store
            from databricks.sdk.service.ml import PublishSpec, PublishSpecPublishMode
            
            publish_spec = PublishSpec(
                online_store=online_store_name,
                online_table_name=f"{prefix}_eventsa45e2a_online",
                publish_mode=PublishSpecPublishMode.CONTINUOUS
            )
            
            publish_result = ws.feature_store.publish_table(
                source_table_name=table_full_name,
                publish_spec=publish_spec
            )
            print(f"✓ Feature store publish initiated: {publish_result}")
            
        except Exception as e2:
            print(f"✗ Error with feature store approach: {e2}")
    
    print("✓ Script completed successfully")

if __name__ == "__main__":
    main()
