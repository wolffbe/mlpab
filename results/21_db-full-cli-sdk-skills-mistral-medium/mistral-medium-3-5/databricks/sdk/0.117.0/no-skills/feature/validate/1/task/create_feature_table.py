#!/usr/bin/env python3

import csv
import json
import os
from databricks.sdk import WorkspaceClient

def main():
    # Environment variables
    schema_name = os.environ.get('MLPAB_DATABRICKS_SCHEMA', 'workspace.mlpab0e0197')
    prefix = os.environ.get('MLPAB_DATABRICKS_PREFIX', 'mlpab0e0197')
    warehouse_id = '4dfab06c923fe3cc'  # The available warehouse
    
    ws = WorkspaceClient()
    
    # Read and filter the CSV data
    valid_categories = {'grocery', 'travel', 'salary', 'rent', 'other'}
    valid_rows = []
    rejected_rows = []
    
    with open('data/events.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_id = row['row_id']
            amount_str = row['amount'].strip()
            category = row['category'].strip()
            
            # Check rule 1: amount is present
            if not amount_str:
                rejected_rows.append(row_id)
                continue
            
            # Check rule 2: amount is within [0, 10000]
            try:
                amount = float(amount_str)
                if amount < 0 or amount > 10000:
                    rejected_rows.append(row_id)
                    continue
            except ValueError:
                rejected_rows.append(row_id)
                continue
            
            # Check rule 3: category is valid
            if category not in valid_categories:
                rejected_rows.append(row_id)
                continue
            
            valid_rows.append(row)
    
    # Write the answers.json file
    os.makedirs('submission', exist_ok=True)
    with open('submission/answers.json', 'w') as f:
        json.dump({"rejected": sorted(rejected_rows)}, f)
    
    print(f"Total rows: {len(valid_rows) + len(rejected_rows)}")
    print(f"Valid rows: {len(valid_rows)}")
    print(f"Rejected rows: {len(rejected_rows)}")
    
    # Extract catalog and schema from schema_name
    catalog_name, schema_part = schema_name.split('.')
    
    # Create the feature table using SQL
    table_name = 'eventsa45e2a'
    full_table_name = f"{schema_name}.{table_name}"
    
    # First, create the table
    create_table_sql = f"""
CREATE TABLE IF NOT EXISTS {full_table_name} (
    row_id STRING,
    account_id STRING,
    event_time BIGINT,
    amount DOUBLE,
    category STRING
) 
USING DELTA
COMMENT 'Feature table for valid events data'
"""
    
    print("Creating table...")
    result = ws.statement_execution.execute_statement(
        statement=create_table_sql,
        warehouse_id=warehouse_id,
        catalog=catalog_name,
        schema=schema_part,
        wait_timeout="30s"
    )
    print(f"Table creation result: {result.status.state}")
    
    # Prepare the INSERT statement with valid data
    insert_values = []
    for row in valid_rows:
        row_id = row['row_id']
        account_id = row['account_id']
        event_time = row['event_time']
        amount = row['amount'].strip()
        category = row['category'].strip()
        
        insert_values.append(f"('{row_id}', '{account_id}', {event_time}, {amount}, '{category}')")
    
    # Batch the inserts to avoid SQL length limits
    batch_size = 50
    for i in range(0, len(insert_values), batch_size):
        batch = insert_values[i:i + batch_size]
        insert_sql = f"""
INSERT INTO {full_table_name} (row_id, account_id, event_time, amount, category)
VALUES {', '.join(batch)}
"""
        print(f"Inserting batch {i//batch_size + 1}...")
        result = ws.statement_execution.execute_statement(
            statement=insert_sql,
            warehouse_id=warehouse_id,
            catalog=catalog_name,
            schema=schema_part,
            wait_timeout="30s"
        )
        print(f"Batch insert result: {result.status.state}")
        if result.status.state != 'SUCCESS':
            print(f"Error: {result.status.error}")
            break
    
    # Now create the online table for low-latency access using feature store
    online_table_name = f"{prefix}_eventsa45e2a_online"
    
    print("Publishing table for online access...")
    try:
        # First try to create an online table directly
        from databricks.sdk.service.catalog import OnlineTable, OnlineTableSpec
        
        online_table_spec = OnlineTableSpec(
            source_table_full_name=full_table_name,
            primary_key_columns=["row_id"],
            timeseries_key="event_time"
        )
        
        online_table = OnlineTable(
            name=online_table_name,
            spec=online_table_spec
        )
        
        created_table = ws.online_tables.create_and_wait(online_table)
        print(f"Online table created: {created_table.name}")
    except Exception as e:
        print(f"Error creating online table: {e}")
        # Try alternative approach using feature store
        try:
            print("Trying feature store publish approach...")
            from databricks.sdk.service.ml import PublishSpec, PublishSpecPublishMode
            
            publish_spec = PublishSpec(
                online_store=f"{catalog_name}.{schema_part}",
                online_table_name=online_table_name,
                publish_mode=PublishSpecPublishMode.SNAPSHOT
            )
            
            feature_table = ws.feature_store.publish_table(
                source_table_name=full_table_name,
                publish_spec=publish_spec
            )
            print(f"Feature table published: {feature_table}")
        except Exception as e2:
            print(f"Error with feature store: {e2}")
    
    print("Done!")

if __name__ == "__main__":
    main()