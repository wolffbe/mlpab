#!/usr/bin/env python3
"""
Script to register a feature table named 'scaleda1a1c9', version 1 on Databricks.
The table contains both training and serving splits standardized using training statistics.
"""

import os
import csv
import io
import time
import databricks.sdk
from databricks.sdk.service.ml import PublishSpec, PublishSpecPublishMode

def read_csv(file_path):
    """Read CSV file and return list of dicts."""
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)

def compute_statistics(train_data):
    """Compute mean and population std for each feature from training data."""
    features = ['f1', 'f2', 'f3', 'f4']
    n = len(train_data)
    
    # Initialize sums
    sums = {f: 0.0 for f in features}
    sum_sq = {f: 0.0 for f in features}
    
    # Compute sums and sum of squares
    for row in train_data:
        for f in features:
            val = float(row[f])
            sums[f] += val
            sum_sq[f] += val * val
    
    # Compute mean and population std
    stats = {}
    for f in features:
        mean = sums[f] / n
        # Population std (no Bessel's correction)
        variance = (sum_sq[f] / n) - (mean * mean)
        std = variance ** 0.5
        stats[f] = {'mean': mean, 'std': std}
    
    return stats

def standardize_value(val, mean, std):
    """Standardize a single value: (x - mean) / std, rounded to 6 decimals."""
    if std == 0:
        return 0.0  # Avoid division by zero
    return round((float(val) - mean) / std, 6)

def standardize_data(data, stats, split_name):
    """Standardize data using precomputed statistics and add split column."""
    features = ['f1', 'f2', 'f3', 'f4']
    standardized = []
    
    for row in data:
        std_row = {
            'row_id': row['row_id'],
            'split': split_name
        }
        for f in features:
            std_row[f] = standardize_value(row[f], stats[f]['mean'], stats[f]['std'])
        standardized.append(std_row)
    
    return standardized

def execute_sql(w, sql, warehouse_id=None):
    """Execute SQL statement and wait for completion."""
    print(f"Executing SQL: {sql[:100]}...")
    
    # Try to find a warehouse
    if warehouse_id is None:
        warehouses = list(w.warehouses.list())
        if warehouses:
            warehouse_id = warehouses[0].id
            print(f"Using warehouse: {warehouse_id}")
    
    result = w.statement_execution.execute_statement(
        statement=sql,
        warehouse_id=warehouse_id,
        catalog=catalog_name,
        schema=schema_part
    )
    
    # Wait for completion
    statement_id = result.statement_id
    print(f"Statement ID: {statement_id}")
    
    # Poll for status
    max_attempts = 120
    for i in range(max_attempts):
        status = w.statement_execution.get_statement(statement_id=statement_id)
        if status.status.state.value in ['SUCCESS', 'FAILED', 'CANCELED']:
            print(f"Statement completed with status: {status.status.state.value}")
            if hasattr(status.status, 'error') and status.status.error:
                print(f"Error: {status.status.error}")
            return status
        time.sleep(2)
    
    print("Timeout waiting for SQL execution")
    return None

def main():
    global catalog_name, schema_part
    
    # Read environment variables
    schema_name = os.environ['MLPAB_DATABRICKS_SCHEMA']
    prefix = os.environ['MLPAB_DATABRICKS_PREFIX']
    
    # Parse schema to get catalog and schema
    # schema_name is like "workspace.mlpab1cf3bf"
    parts = schema_name.split('.')
    catalog_name = parts[0]
    schema_part = '.'.join(parts[1:])
    
    print(f"Catalog: {catalog_name}")
    print(f"Schema: {schema_part}")
    print(f"Prefix: {prefix}")
    
    # Read data
    print("Reading training data...")
    train_data = read_csv('data/features_train.csv')
    print(f"Training rows: {len(train_data)}")
    
    print("Reading serving data...")
    serve_data = read_csv('data/features_serve.csv')
    print(f"Serving rows: {len(serve_data)}")
    
    # Compute statistics from training data only
    print("Computing statistics from training data...")
    stats = compute_statistics(train_data)
    print(f"Statistics computed")
    
    # Standardize both datasets
    print("Standardizing training data...")
    train_std = standardize_data(train_data, stats, 'train')
    
    print("Standardizing serving data...")
    serve_std = standardize_data(serve_data, stats, 'serve')
    
    # Connect to Databricks
    print("Connecting to Databricks...")
    w = databricks.sdk.WorkspaceClient()
    
    # Create the table using SQL
    full_table_name = f"{catalog_name}.{schema_part}.scaleda1a1c9"
    
    print(f"\nCreating table {full_table_name}...")
    
    # Create the table using CREATE TABLE statement
    create_table_sql = f"""
CREATE TABLE IF NOT EXISTS {full_table_name} (
    row_id STRING,
    split STRING,
    f1 DOUBLE,
    f2 DOUBLE,
    f3 DOUBLE,
    f4 DOUBLE
) USING DELTA
"""
    
    execute_sql(w, create_table_sql)
    print(f"Table {full_table_name} created")
    
    # Insert data using INSERT INTO with VALUES
    # We'll batch the inserts to avoid very long SQL statements
    batch_size = 200
    all_data = train_std + serve_std
    
    print(f"\nInserting {len(all_data)} rows in batches of {batch_size}...")
    
    for i in range(0, len(all_data), batch_size):
        batch = all_data[i:i+batch_size]
        values_clauses = []
        for row in batch:
            values_clauses.append(
                f"('{row['row_id']}', '{row['split']}', {row['f1']}, {row['f2']}, {row['f3']}, {row['f4']})"
            )
        
        insert_sql = f"""
INSERT INTO {full_table_name} (row_id, split, f1, f2, f3, f4)
VALUES {', '.join(values_clauses)}
"""
        
        print(f"  Inserting batch {i//batch_size + 1}/{(len(all_data) + batch_size - 1)//batch_size}...")
        execute_sql(w, insert_sql)
    
    print("All data inserted")
    
    # Verify the data
    count_sql = f"SELECT COUNT(*) as cnt FROM {full_table_name}"
    result = execute_sql(w, count_sql)
    print(f"Table row count verification complete")
    
    # Now create an online store for low-latency lookup
    online_store_name = f"{prefix}_scaleda1a1c9_store"
    print(f"\nCreating online store: {online_store_name}")
    
    try:
        online_store = w.feature_store.create_online_store(
            name=online_store_name,
            capacity="SMALL"
        )
        print(f"Online store created: {online_store}")
    except Exception as e:
        print(f"Error creating online store: {e}")
        # Maybe it already exists, try to get it
        try:
            online_store = w.feature_store.get_online_store(online_store_name)
            print(f"Online store already exists: {online_store}")
        except Exception as e2:
            print(f"Cannot get existing online store: {e2}")
            raise
    
    # Publish the table to the online store
    online_table_name = f"{prefix}_scaleda1a1c9_online"
    print(f"\nPublishing table to online store as: {online_table_name}")
    
    try:
        publish_response = w.feature_store.publish_table(
            source_table_name=full_table_name,
            publish_spec=PublishSpec(
                online_store=online_store_name,
                online_table_name=online_table_name,
                publish_mode=PublishSpecPublishMode.OVERWRITE
            )
        )
        print(f"Publish response: {publish_response}")
    except Exception as e:
        print(f"Error publishing table: {e}")
        raise
    
    print("\n=== SUCCESS ===")
    print(f"Feature table 'scaleda1a1c9' version 1 registered in {catalog_name}.{schema_part}")
    print(f"Online table '{online_table_name}' published to store '{online_store_name}'")

if __name__ == '__main__':
    main()
