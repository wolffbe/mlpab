#!/usr/bin/env python3
"""
Script to register a feature table.
"""

import os
import csv
import time
import databricks.sdk
from databricks.sdk.service.ml import PublishSpec, PublishSpecPublishMode

def read_csv(file_path):
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)

def compute_statistics(train_data):
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
    if std == 0:
        return 0.0
    return round((float(val) - mean) / std, 6)

def standardize_data(data, stats, split_name):
    features = ['f1', 'f2', 'f3', 'f4']
    standardized = []
    for row in data:
        std_row = {'row_id': row['row_id'], 'split': split_name}
        for f in features:
            std_row[f] = standardize_value(row[f], stats[f]['mean'], stats[f]['std'])
        standardized.append(std_row)
    return standardized

def exec_sql(w, sql, warehouse_id):
    """Execute SQL and wait for completion."""
    print(f"Executing: {sql[:80]}...")
    result = w.statement_execution.execute_statement(
        statement=sql,
        warehouse_id=warehouse_id,
        catalog=catalog_name,
        schema=schema_part
    )
    statement_id = result.statement_id
    for _ in range(120):
        status = w.statement_execution.get_statement(statement_id=statement_id)
        if status.status.state.value in ['SUCCESS', 'FAILED', 'CANCELED']:
            if status.status.state.value == 'FAILED':
                print(f"FAILED: {status.status.error}")
                raise Exception(f"SQL failed: {status.status.error}")
            return
        time.sleep(2)
    raise Exception("Timeout waiting for SQL")

def main():
    global catalog_name, schema_part
    
    schema_name = os.environ['MLPAB_DATABRICKS_SCHEMA']
    prefix = os.environ['MLPAB_DATABRICKS_PREFIX']
    parts = schema_name.split('.')
    catalog_name = parts[0]
    schema_part = '.'.join(parts[1:])
    
    print(f"Catalog: {catalog_name}, Schema: {schema_part}")
    
    # Read and process data
    train_data = read_csv('data/features_train.csv')
    serve_data = read_csv('data/features_serve.csv')
    stats = compute_statistics(train_data)
    train_std = standardize_data(train_data, stats, 'train')
    serve_std = standardize_data(serve_data, stats, 'serve')
    all_data = train_std + serve_std
    print(f"Total rows: {len(all_data)}")
    
    w = databricks.sdk.WorkspaceClient()
    warehouses = list(w.warehouses.list())
    warehouse_id = warehouses[0].id if warehouses else None
    print(f"Warehouse: {warehouse_id}")
    
    full_table_name = f"{catalog_name}.{schema_part}.scaleda1a1c9"
    
    # Create table
    print("Creating table...")
    exec_sql(w, f"CREATE TABLE IF NOT EXISTS {full_table_name} (row_id STRING, split STRING, f1 DOUBLE, f2 DOUBLE, f3 DOUBLE, f4 DOUBLE) USING DELTA", warehouse_id)
    
    # Insert all data in one batch
    print("Inserting data...")
    values = []
    for row in all_data:
        values.append(f"('{row['row_id']}', '{row['split']}', {row['f1']}, {row['f2']}, {row['f3']}, {row['f4']})")
    
    insert_sql = f"INSERT INTO {full_table_name} (row_id, split, f1, f2, f3, f4) VALUES {', '.join(values)}"
    exec_sql(w, insert_sql, warehouse_id)
    
    print("Data inserted")
    
    # Create online store and publish
    online_store_name = f"{prefix}_scaleda1a1c9_store"
    online_table_name = f"{prefix}_scaleda1a1c9_online"
    
    print("Creating online store...")
    try:
        w.feature_store.create_online_store(name=online_store_name, capacity="SMALL")
    except Exception as e:
        print(f"Online store may exist: {e}")
    
    print("Publishing table...")
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
