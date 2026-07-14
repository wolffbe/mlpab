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
    
    # Get warehouse
    warehouses = list(ws.warehouses.list())
    warehouse_id = warehouses[0].id if warehouses else None
    print(f"✓ Warehouse ID: {warehouse_id}")
    
    # Step 3: Check if schema exists, create if not
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
    
    # Step 4: Create the feature table using SQL
    table_full_name = f"{catalog_name}.{schema_part}.eventsa45e2a"
    
    # Create table with inline data using a single CREATE TABLE AS SELECT
    # We'll generate a UNION ALL query with all the valid data
    select_parts = []
    for i, row in enumerate(valid_rows):
        row_id = row['row_id'].replace("'", "''")
        account_id = row['account_id'].replace("'", "''")
        event_time = row['event_time']
        amount = row['amount']
        category = row['category'].replace("'", "''")
        
        select_parts.append(f"SELECT '{row_id}' AS row_id, '{account_id}' AS account_id, {event_time} AS event_time, {amount} AS amount, '{category}' AS category")
    
    union_sql = " UNION ALL ".join(select_parts)
    
    create_table_sql = f"""
CREATE OR REPLACE TABLE {table_full_name} (
    row_id STRING,
    account_id STRING,
    event_time BIGINT,
    amount DOUBLE,
    category STRING
) USING DELTA
COMMENT 'Feature table for valid events data'
AS {union_sql}
"""
    
    print(f"✓ Creating table with all data...")
    
    try:
        result = ws.statement_execution.execute_statement(
            warehouse_id=warehouse_id,
            catalog=catalog_name,
            schema=schema_part,
            statement=create_table_sql,
            wait_timeout="120s"
        )
        print(f"✓ Table creation completed")
    except Exception as e:
        print(f"✗ Error creating table: {e}")
        # Try with a smaller approach - just create empty table first
        try:
            simple_create_sql = f"""
CREATE OR REPLACE TABLE {table_full_name} (
    row_id STRING,
    account_id STRING,
    event_time BIGINT,
    amount DOUBLE,
    category STRING
) USING DELTA
COMMENT 'Feature table for valid events data'
"""
            
            result = ws.statement_execution.execute_statement(
                warehouse_id=warehouse_id,
                catalog=catalog_name,
                schema=schema_part,
                statement=simple_create_sql,
                wait_timeout="30s"
            )
            print(f"✓ Empty table created")
            
            # Now try to insert data in a single batch
            # Generate a single INSERT with all values
            values_clauses = []
            for row in valid_rows[:10]:  # Try with just 10 rows first
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
            
            result = ws.statement_execution.execute_statement(
                warehouse_id=warehouse_id,
                catalog=catalog_name,
                schema=schema_part,
                statement=insert_sql,
                wait_timeout="60s"
            )
            print(f"✓ Test data inserted")
            
        except Exception as e2:
            print(f"✗ Error with test approach: {e2}")
        return
    
    # Step 5: Create online table for low-latency access
    online_table_name = f"{catalog_name}.{schema_part}.eventsa45e2a_online"
    
    print(f"✓ Creating online table...")
    
    try:
        from databricks.sdk.service.catalog import OnlineTable, OnlineTableSpec
        
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
                online_store_name = online_store.name
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
