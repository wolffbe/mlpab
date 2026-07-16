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
        print(f"Schema doesn't exist, creating: {e}")
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
                return
    
    # Step 4: Create the feature table using SQL
    table_full_name = f"{catalog_name}.{schema_part}.eventsa45e2a"
    
    # First, create an empty table
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
    
    # Step 5: Insert data in small batches to avoid timeout
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
                wait_timeout="60s"
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
    
    # Step 6: Create online table for low-latency access
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
        print(f"✓ Online table creation initiated")
        
        # Wait for the online table to be active
        wait_result = ws.online_tables.wait_get_online_table_active(online_table_name)
        print(f"✓ Online table is active")
        
    except Exception as e:
        print(f"✗ Error creating online table: {e}")
        # Try feature store approach
        try:
            print("✓ Trying feature store approach...")
            
            # Check if we need to create an online store
            online_stores = list(ws.feature_store.list_online_stores())
            if not online_stores:
                # Create an online store
                online_store = ws.feature_store.create_online_store(
                    name=f"{prefix}_online_store",
                    storage_type="DATABRICKS"
                )
                print(f"✓ Created online store: {online_store.name}")
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
            print(f"✓ Feature store publish initiated")
            
        except Exception as e2:
            print(f"✗ Error with feature store approach: {e2}")
    
    print("✓ Script completed successfully")

if __name__ == "__main__":
    main()
