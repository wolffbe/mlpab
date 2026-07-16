#!/usr/bin/env python3
"""
Simplified script to register a feature table.
"""

import os
import csv
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
    
    sums = {f: 0.0 for f in features}
    sum_sq = {f: 0.0 for f in features}
    
    for row in train_data:
        for f in features:
            val = float(row[f])
            sums[f] += val
            sum_sq[f] += val * val
    
    stats = {}
    for f in features:
        mean = sums[f] / n
        variance = (sum_sq[f] / n) - (mean * mean)
        std = variance ** 0.5
        stats[f] = {'mean': mean, 'std': std}
    
    return stats

def standardize_value(val, mean, std):
    """Standardize a single value: (x - mean) / std, rounded to 6 decimals."""
    if std == 0:
        return 0.0
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

def main():
    # Read environment variables
    schema_name = os.environ['MLPAB_DATABRICKS_SCHEMA']
    prefix = os.environ['MLPAB_DATABRICKS_PREFIX']
    
    parts = schema_name.split('.')
    catalog_name = parts[0]
    schema_part = '.'.join(parts[1:])
    
    print(f"Catalog: {catalog_name}, Schema: {schema_part}, Prefix: {prefix}")
    
    # Read and process data
    print("Reading and processing data...")
    train_data = read_csv('data/features_train.csv')
    serve_data = read_csv('data/features_serve.csv')
    
    stats = compute_statistics(train_data)
    train_std = standardize_data(train_data, stats, 'train')
    serve_std = standardize_data(serve_data, stats, 'serve')
    
    all_data = train_std + serve_std
    print(f"Total rows: {len(all_data)}")
    
    # Connect to Databricks
    w = databricks.sdk.WorkspaceClient()
    
    # Get warehouse
    warehouses = list(w.warehouses.list())
    warehouse_id = warehouses[0].id if warehouses else None
    print(f"Using warehouse: {warehouse_id}")
    
    # Create table
    full_table_name = f"{catalog_name}.{schema_part}.scaleda1a1c9"
    
    # Build a single CREATE TABLE AS SELECT statement with all data
    # This is more efficient than multiple INSERT statements
    values_clauses = []
    for row in all_data:
        values_clauses.append(
            f"('{row['row_id']}', '{row['split']}', {row['f1']}, {row['f2']}, {row['f3']}, {row['f4']})"
        )
    
    # Split into batches to avoid SQL statement being too long
    batch_size = 100
    for i in range(0, len(values_clauses), batch_size):
        batch = values_clauses[i:i+batch_size]
        
        if i == 0:
            # First batch: create table
            sql = f"""
CREATE OR REPLACE TABLE {full_table_name} 
USING DELTA
AS SELECT * FROM (
    SELECT 
        'row_id' as row_id, 'split' as split, 0.0 as f1, 0.0 as f2, 0.0 as f3, 0.0 as f4 
    WHERE FALSE
    UNION ALL
    SELECT * FROM (
        VALUES {', '.join(batch)}
    ) AS t(row_id, split, f1, f2, f3, f4)
)
"""
        else:
            # Subsequent batches: insert into table
            sql = f"""
INSERT INTO {full_table_name} (row_id, split, f1, f2, f3, f4)
SELECT * FROM (
    VALUES {', '.join(batch)}
) AS t(row_id, split, f1, f2, f3, f4)
"""
        
        print(f"Executing batch {i//batch_size + 1}/{(len(values_clauses) + batch_size - 1)//batch_size}...")
        
        result = w.statement_execution.execute_statement(
            statement=sql,
            warehouse_id=warehouse_id,
            catalog=catalog_name,
            schema=schema_part
        )
        
        statement_id = result.statement_id
        
        # Wait for completion
        max_attempts = 60
        for j in range(max_attempts):
            status = w.statement_execution.get_statement(statement_id=statement_id)
            if status.status.state.value in ['SUCCESS', 'FAILED', 'CANCELED']:
                if status.status.state.value == 'FAILED':
                    print(f"Error: {status.status.error}")
                    raise Exception(f"SQL execution failed: {status.status.error}")
                break
            if j % 10 == 0:
                print(f"  Waiting for statement {statement_id}...")
            import time
            time.sleep(2)
    
    print("Table created and data loaded")
    
    # Create online store and publish
    online_store_name = f"{prefix}_scaleda1a1c9_store"
    online_table_name = f"{prefix}_scaleda1a1c9_online"
    
    print(f"\nCreating online store: {online_store_name}")
    try:
        w.feature_store.create_online_store(name=online_store_name, capacity="SMALL")
    except Exception as e:
        print(f"Online store may already exist: {e}")
    
    print(f"Publishing table to online store...")
    w.feature_store.publish_table(
        source_table_name=full_table_name,
        publish_spec=PublishSpec(
            online_store=online_store_name,
            online_table_name=online_table_name,
            publish_mode=PublishSpecPublishMode.OVERWRITE
        )
    )
    
    print("\n=== SUCCESS ===")
    print(f"Feature table 'scaleda1a1c9' version 1 registered in {catalog_name}.{schema_part}")
    print(f"Online table '{online_table_name}' published to store '{online_store_name}'")

if __name__ == '__main__':
    main()
